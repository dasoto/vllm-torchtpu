# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shortest-remaining-first ordering and hold-back of running prefill chunks.

vLLM schedules RUNNING in admission order before WAITING, so an in-progress
long prefill chunk can take the step's token budget before a short waiting
request is reached. Each step (FCFS policy, prefills not throttled) this
scheduler:

* orders RUNNING as decodes first, in their existing order, then prefill
  chunks by remaining prompt tokens, fewest first, with stalled chunks (below)
  ahead of the rest;
* holds back, for this step, the longest running prefill chunks needed for the
  request upstream admits next (the first admissible request in
  ``kv_holding_waiting``, then ``waiting``) to fit in the remaining budget,
  when that request has fewer uncached tokens and free KV blocks cover the
  kept requests plus it, so a hold does not cause preemption.

Held chunks keep their KV blocks; ``max_num_running_reqs`` and
``max_num_active_reqs`` are lowered by the number held during the step so
admission never exceeds ``max_num_seqs``. A hold whose target is not
admitted backs off exponentially for that target.

Anti-starvation: a running prefill chunk that goes
``VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS`` steps in a row without getting at least
one KV block of tokens in a step, whether it was held or crowded out by shorter
prefills, is stalled: it moves ahead of the other prefill chunks and is not
held until it gets a block. Decodes still go first, so a stalled chunk can
still wait while decodes fill the budget.

With ``VLLM_TPU_SRPT_ATTN_WORK_CAP=W`` (default 0, off) a step's attention work,
``sum(n * (c + n / 2))`` over the chunks it schedules (``n`` new tokens after
``c`` computed ones, decodes included), is kept under ``W`` by shrinking prefill
chunks. Under expert-parallel DP lockstep, one rank's deep-context chunk
otherwise stretches every rank's step and the decodes in it. A shrunk chunk's
end, not its length, is rounded down to a multiple of
``VLLM_TPU_SRPT_ATTN_TILE_TOKENS`` (default 32, the query tile of the TPU
attention kernels), so a chunk that starts off that grid still ends on it. The
step's first prefill chunk always advances to the next multiple, so a chunk
whose context alone exceeds ``W`` still advances. A chunk the cap shrank counts
as progress for the anti-starvation rule above.

The cap only protects decodes, so it applies while this rank decodes in the
step being scheduled or, under EP lockstep (``DP > 1`` and expert parallel),
while another DP rank decoded in the last step the runner reported
(``TPUModelRunner`` adds the highest and the lowest decoding rank to the DP
metadata all-reduce it runs under ``_dp_lockstep_enabled``; with plain DP,
``_other_rank_decoding`` stays ``False`` and the cap applies only while this
rank decodes). While no rank decodes, prefill chunks are not shrunk.

Under KV pressure upstream preempts ``running[-1]``, which is the longest
non-stalled prefill chunk, so a long prompt can be preempted repeatedly.

Select with ``--scheduler-cls``
``vllm_torchtpu.core.experimental.srpt_prefill_scheduler.SrptPrefillScheduler``.
"""

import math
import time
from collections import OrderedDict
from itertools import chain, islice
from typing import Any, cast

from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.metrics.stats import KV_FETCH_WAITING_TO_START
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_MAX_BACKOFF_SHIFT = 6
_ATTN_CAP_LOG_EVERY = 10000
_WARM_WINDOW = 128
_WARM_STARVATION_SECONDS = 90.0
_HARD_STARVATION_SECONDS = 600.0
_CRITICAL_HOLD_SECONDS = 900.0
_WARM_FOLLOWUP_MIN_HIT_TOKENS = 8192
_WARM_FOLLOWUP_MAX_UNCACHED_TOKENS = 32768
_MAX_EXTERNAL_BLOCK_HASHES = 65536


class SrptPrefillScheduler(AsyncScheduler):
    """Order RUNNING prefill chunks by fewest remaining tokens and hold back
    longer ones for a shorter waiting request."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_consecutive_holds = envs.VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS
        # Running prefill request id -> consecutive steps with less than one
        # block of progress.
        self._stalled_steps: dict[str, int] = {}
        # Target request id -> (failed holds, first step a hold may retry).
        self._hold_backoff: dict[str, tuple[int, int]] = {}
        # Bounded LRU directory of block hashes offloaded to external KV tiers
        # (e.g. Raiden Host DRAM) so warm sessions evicted from HBM are still
        # recognized as short-uncached follow-ups without touching connector state.
        self._external_block_hashes: OrderedDict[Any, None] = OrderedDict()
        self.attn_work_cap = envs.VLLM_TPU_SRPT_ATTN_WORK_CAP
        self.attn_tile_tokens = max(envs.VLLM_TPU_SRPT_ATTN_TILE_TOKENS, 1)
        self.cadence_no_escape = envs.VLLM_TPU_PREFILL_CADENCE_NO_ESCAPE
        self.promote_warm_waiting = envs.VLLM_TPU_SRPT_PROMOTE_WARM_WAITING
        # Whether another DP rank decoded in the last step the runner reported,
        # and whether the cap applies to the step being scheduled.
        self._other_rank_decoding = False
        self._attn_cap_on = False
        # Attention work and whether a prefill chunk was scheduled so far in
        # this step's Scheduler.schedule(), and the requests the cap shrank.
        self._in_schedule = False
        self._attn_work = 0.0
        self._attn_prefill_scheduled = False
        self._attn_capped: set[str] = set()
        self._attn_steps = 0
        self._attn_cap_on_steps = 0
        self._attn_capped_chunks = 0
        logger.info(
            "SrptPrefillScheduler active (max_consecutive_holds=%d, "
            "attn_work_cap=%d, attn_tile_tokens=%d, cadence_no_escape=%s, "
            "promote_warm_waiting=%s)",
            self.max_consecutive_holds,
            self.attn_work_cap,
            self.attn_tile_tokens,
            self.cadence_no_escape,
            self.promote_warm_waiting,
        )

    def _record_external_block_hashes(self, output: SchedulerOutput) -> None:
        if self.connector is None or not output.num_scheduled_tokens:
            return
        tokens_per_hash = max(int(self.cache_config.block_size or 1), 1)
        ext = self._external_block_hashes
        for req_id, n_sched in output.num_scheduled_tokens.items():
            if n_sched <= 0:
                continue
            req = self.requests.get(req_id)
            if req is None or not req.block_hashes:
                continue
            n_full = min(
                len(req.block_hashes), req.num_computed_tokens // tokens_per_hash
            )
            for h in req.block_hashes[:n_full]:
                if h in ext:
                    ext.move_to_end(h)
                else:
                    if len(ext) >= _MAX_EXTERNAL_BLOCK_HASHES:
                        ext.popitem(last=False)
                    ext[h] = None

    def _estimated_prefix_hit_tokens(
        self, request: Request, local_hit_tokens: int | None = None
    ) -> int:
        if local_hit_tokens is None:
            _, local_hit_tokens = self._prefix_hit(request)
        if (
            self.connector is None
            or not self._external_block_hashes
            or not request.block_hashes
            or request.num_computed_tokens > 0
        ):
            return local_hit_tokens
        ext = self._external_block_hashes
        matched_blocks = 0
        for h in request.block_hashes:
            if h not in ext:
                break
            matched_blocks += 1
        if matched_blocks <= 0:
            return local_hit_tokens
        tokens_per_hash = max(int(self.cache_config.block_size or 1), 1)
        ext_tokens = min(
            matched_blocks * tokens_per_hash,
            max(request.num_prompt_tokens - 1, 0),
        )
        conn_sched = getattr(self.connector, "connector_scheduler", None) or getattr(
            self.connector, "_scheduler", None
        )
        conn_cfg = getattr(conn_sched, "config", None)
        if getattr(conn_cfg, "supports_partial_tail", False):
            align = getattr(conn_cfg, "tokens_per_hash", None)
        else:
            align = getattr(conn_cfg, "alignment_tokens", None)
        if isinstance(align, int) and align > 0:
            ext_tokens = (ext_tokens // align) * align
        return max(local_hit_tokens, ext_tokens)

    def _waiting_req_fits_kv(
        self, request: Request, free_blocks: int, reserved_async_blocks: int
    ) -> tuple[bool, int, int]:
        """Return ``(fits, local_hit, est_hit)`` matching ``allocate_slots``."""
        hit_blocks, local_hit = self._prefix_hit(request)
        est_hit = self._estimated_prefix_hit_tokens(request, local_hit)
        is_async_ext = est_hit > local_hit
        if not self.scheduler_reserve_full_isl and not is_async_ext:
            return True, local_hit, est_hit
        if is_async_ext:
            block_aligned_local = (local_hit // self.block_size) * self.block_size
            computed_blocks = self.kv_cache_manager.truncate_computed_blocks(
                hit_blocks, block_aligned_local
            ).blocks
            num_local = block_aligned_local
            total_comp = min(est_hit, self.max_model_len)
        else:
            computed_blocks = hit_blocks.blocks
            num_local = local_hit
            total_comp = min(local_hit, self.max_model_len)
        watermark = self.kv_cache_manager.watermark_blocks if self.running else 0
        prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
        if self.scheduler_reserve_full_isl:
            full_num_tokens = min(request.num_tokens, self.max_model_len)
            full_needed = self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=full_num_tokens,
                new_computed_blocks=computed_blocks,
                num_encoder_tokens=0,
                total_computed_tokens=total_comp,
                num_local_computed_tokens=num_local,
                num_tokens_main_model=full_num_tokens,
                apply_admission_cap=True,
                prefill_end=prefill_end,
            )
            if full_needed + watermark > free_blocks:
                return False, local_hit, est_hit
        if is_async_ext:
            step_needed = self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=total_comp,
                new_computed_blocks=computed_blocks,
                num_encoder_tokens=0,
                total_computed_tokens=total_comp,
                num_local_computed_tokens=num_local,
                num_tokens_main_model=total_comp,
                prefill_end=prefill_end,
            )
            if step_needed + watermark > free_blocks - reserved_async_blocks:
                return False, local_hit, est_hit
        return True, local_hit, est_hit

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        ok = super().reset_prefix_cache(
            reset_running_requests=reset_running_requests,
            reset_connector=reset_connector,
        )
        if ok and reset_connector:
            self._external_block_hashes.clear()
        return ok

    def _maybe_promote_warm_waiting(self, throttle_prefills: bool = False) -> None:
        """Promote fitting ready requests ahead of blocked ones in SRPT order."""
        if (
            not self.promote_warm_waiting
            or self.policy != SchedulingPolicy.FCFS
            or len(self.waiting) < 2
        ):
            return
        ready = [
            r
            for r in islice(self.waiting, _WARM_WINDOW)
            if not self._is_blocked_waiting_status(r.status)
        ]
        if len(ready) < 2:
            return
        now = time.time()
        kv_fetch_stages = getattr(self, "_kv_fetch_stages", None) or {}
        has_inflight_prefills = bool(getattr(self, "_inflight_prefills", None))
        free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
        reserved_async_blocks = (
            self._inflight_prefill_reserved_blocks() + self._spec_decode_step_blocks
        )
        key_cache: dict[str, tuple[int, int, int, int, float, float]] = {}

        def rank_key(req: Request) -> tuple[int, int, int, int, float, float]:
            cached_key = key_cache.get(req.request_id)
            if cached_key is not None:
                return cached_key
            fits, local_hit, est_hit = self._waiting_req_fits_kv(
                req, free_blocks, reserved_async_blocks
            )
            uncached = max(req.num_tokens - est_hit, 0)
            wait_s = now - req.arrival_time
            is_stalled_fetch = kv_fetch_stages.get(req) == KV_FETCH_WAITING_TO_START
            if wait_s > _HARD_STARVATION_SECONDS:
                tier = -1
            elif (
                req.num_preemptions > 0
                or uncached <= 8192
                or wait_s > _WARM_STARVATION_SECONDS
            ):
                tier = 0
            elif (
                est_hit >= _WARM_FOLLOWUP_MIN_HIT_TOKENS
                and uncached <= _WARM_FOLLOWUP_MAX_UNCACHED_TOKENS
            ):
                tier = 1
            else:
                tier = 2
            # Upstream Scheduler.schedule() executes `break` (head-of-line block)
            # when waiting[0] fails KV allocation. Demote non-critical requests
            # that cannot currently pass allocate_slots behind every request that
            # CAN fit, while letting critically aged requests hold the head so
            # HBM blocks drain for them before timeout.
            exceeds_hbm = (
                1 if (wait_s <= _CRITICAL_HOLD_SECONDS and not fits) else 0
            )
            demote_stalled_fetch = (
                1 if (is_stalled_fetch and not has_inflight_prefills) else 0
            )
            # On decode-only cadence steps (throttle_prefills=True), non-external
            # prefills hit `break` at defer_prefills whereas external async loads
            # (`load_kv_async=True`, 0 compute tokens) can launch Host->HBM DMA.
            not_async_ext = 0 if (throttle_prefills and est_hit > local_hit) else 1
            primary_metric = req.arrival_time if tier == -1 else float(uncached)
            res = (
                exceeds_hbm,
                not_async_ext,
                tier,
                demote_stalled_fetch,
                primary_metric,
                req.arrival_time,
            )
            key_cache[req.request_id] = res
            return res

        queue = cast(FCFSRequestQueue, self.waiting)
        fitting = [r for r in ready if rank_key(r)[0] == 0]
        if fitting:
            ordered = sorted(fitting, key=rank_key)
            if ready[: len(ordered)] != ordered:
                for req in ordered:
                    queue.remove(req)
                for req in reversed(ordered):
                    self.waiting.prepend_request(req)
            return

        head_req = ready[0]
        head_key = rank_key(head_req)
        best_req = min(ready, key=rank_key)
        if best_req is not head_req and rank_key(best_req) < head_key:
            queue.remove(best_req)
            self.waiting.prepend_request(best_req)

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        local_decoding = any(not r.is_prefill_chunk for r in self.running)
        # The cap only protects decodes: it applies while this rank decodes in
        # this step or another rank decoded in the last reported step.
        self._attn_cap_on = bool(self.attn_work_cap) and (
            self._other_rank_decoding or local_decoding
        )
        if self.cadence_no_escape:
            self.prefill_capacity_bound = False
        self._maybe_promote_warm_waiting(throttle_prefills=throttle_prefills)

        held: list[Request] = []
        stashed_kv_holding: list[Request] = []
        target = None
        prefilling: list[Request] = []
        prev_pause_state = self._pause_state
        prev_max_scheduled = self.max_num_scheduled_tokens
        cross_rank_decode_hold = (
            throttle_prefills
            and self.cadence_no_escape
            and not local_decoding
            and self._other_rank_decoding
        )
        if cross_rank_decode_hold:
            held = [r for r in self.running if r.is_prefill_chunk]
            if prev_pause_state == PauseState.UNPAUSED:
                self._pause_state = PauseState.PAUSED_NEW
            self.max_num_scheduled_tokens = min(
                prev_max_scheduled, self.scheduler_config.max_num_seqs
            )
        elif not throttle_prefills and self.policy == SchedulingPolicy.FCFS:
            self._prune_hold_state()
            self._order_running()
            prefilling = [r for r in self.running if r.is_prefill_chunk]
            held, target = self._select_chunks_to_hold()

        if (
            throttle_prefills
            and not cross_rank_decode_hold
            and self.connector is not None
            and self.kv_holding_waiting
        ):
            # Completed async KV loads waiting for a prefill step sit in
            # kv_holding_waiting and trigger `break` at `defer_prefills`,
            # preventing `self.waiting` from starting new 0-compute async DMA
            # loads during decode steps. Temporarily step past them while
            # keeping WAITING_FOR_REMOTE_KVS entries visible for poll completion.
            keep_kv_holding: list[Request] = []
            while self.kv_holding_waiting:
                req_k = self.kv_holding_waiting.pop_request()
                if (
                    req_k.status == RequestStatus.WAITING
                    and 0 < req_k.num_computed_tokens < req_k.num_tokens - 1
                ):
                    stashed_kv_holding.append(req_k)
                else:
                    keep_kv_holding.append(req_k)
            if keep_kv_holding:
                self.kv_holding_waiting.prepend_requests(keep_kv_holding)

        held_ids = {r.request_id for r in held}
        self.running[:] = [r for r in self.running if r.request_id not in held_ids]
        self.max_num_running_reqs -= len(held)
        self.max_num_active_reqs -= len(held)
        self._attn_work = 0.0
        self._attn_prefill_scheduled = False
        self._attn_capped.clear()
        self._in_schedule = True
        try:
            output = super().schedule(throttle_prefills)
            if self.cadence_no_escape:
                self.prefill_capacity_bound = False
        finally:
            self._in_schedule = False
            self._pause_state = prev_pause_state
            self.max_num_scheduled_tokens = prev_max_scheduled
            self.max_num_running_reqs += len(held)
            self.max_num_active_reqs += len(held)
            self.running.extend(held)
            if stashed_kv_holding:
                self.kv_holding_waiting.prepend_requests(stashed_kv_holding)

        self._record_external_block_hashes(output)

        # One KV cache block of the main attention group; not self.block_size,
        # which for hybrid models is the LCM of all group block sizes and can
        # exceed a whole chunk. When DCP or unified hybrid pooling also scales
        # cache_config.block_size above the per-step chunk budget, measure
        # forward progress at half a chunk so running cold prefills that get
        # their full chunk budget are not falsely marked stalled.
        max_chunk = self.scheduler_config.max_num_batched_tokens
        if 0 < self.scheduler_config.long_prefill_token_threshold < max_chunk:
            max_chunk = self.scheduler_config.long_prefill_token_threshold
        progress_tokens = (
            self.cache_config.block_size
            if self.cache_config.block_size <= max_chunk
            else max(1, max_chunk // 2)
        )
        for r in prefilling:
            n = output.num_scheduled_tokens.get(r.request_id, 0)
            if (
                n >= progress_tokens
                # Got all the attention work cap allowed this step.
                or (n > 0 and r.request_id in self._attn_capped)
                or not r.is_prefill_chunk
            ):
                self._stalled_steps.pop(r.request_id, None)
            else:
                self._stalled_steps[r.request_id] = (
                    self._stalled_steps.get(r.request_id, 0) + 1
                )
        if target is not None:
            self._record_hold_outcome(target)
        if self.attn_work_cap:
            self._attn_steps += 1
            self._attn_cap_on_steps += self._attn_cap_on
            self._attn_capped_chunks += len(self._attn_capped)
            if self._attn_steps % _ATTN_CAP_LOG_EVERY == 0:
                logger.info(
                    "SrptPrefillScheduler attention work cap: steps=%d "
                    "cap_on_steps=%d capped_chunks=%d",
                    self._attn_steps,
                    self._attn_cap_on_steps,
                    self._attn_capped_chunks,
                )
        return output

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        # TPUModelRunner sets this on the outputs of the steps it samples; the
        # shared empty output of a step without tokens and idle DP dummy steps
        # produce no output, so the last report stands until the next sampled
        # step on this rank.
        decoding = getattr(model_runner_output, "other_rank_decoding", None)
        if decoding is not None:
            self._other_rank_decoding = decoding
        return super().update_from_output(scheduler_output, model_runner_output)

    def _attn_cap_tokens(
        self, context: int, num_new_tokens: int, work: float, prefill_scheduled: bool
    ) -> int:
        """Tokens of a prefill chunk of ``num_new_tokens`` after ``context``
        computed ones that fit the attention work left after ``work``, with
        the chunk's end rounded down to the attention tile."""
        room = self.attn_work_cap - work
        fit = int(math.sqrt(context * context + 2 * room) - context) if room > 0 else 0
        if fit >= num_new_tokens:
            return num_new_tokens
        # Rounding the end rather than the length lets a chunk that starts off
        # the tile grid end on it; rounding down keeps the step under the cap.
        tile = self.attn_tile_tokens
        end = (context + fit) // tile * tile
        if end > context:
            return end - context
        if not prefill_scheduled:
            return min(num_new_tokens, self._tokens_to_next_tile(context))
        return 0

    def _tokens_to_next_tile(self, context: int) -> int:
        """Tokens from ``context`` to the next multiple of the attention tile."""
        return self.attn_tile_tokens - context % self.attn_tile_tokens

    def _reserve_prefill_lookahead(
        self, request: Request, num_computed_tokens: int, num_new_tokens: int
    ) -> int:
        # Scheduler.schedule() calls this on every request it is about to
        # schedule, in both queues, after its token budget limits, so it is
        # where the step's attention work is tracked and capped.
        if not (self._in_schedule and self._attn_cap_on):
            return super()._reserve_prefill_lookahead(
                request, num_computed_tokens, num_new_tokens
            )
        prefill = num_computed_tokens < request.num_prompt_tokens
        if prefill and num_new_tokens > 0:
            capped = self._attn_cap_tokens(
                num_computed_tokens,
                num_new_tokens,
                self._attn_work,
                self._attn_prefill_scheduled,
            )
            if capped < num_new_tokens:
                self._attn_capped.add(request.request_id)
                num_new_tokens = capped
        num_new_tokens = super()._reserve_prefill_lookahead(
            request, num_computed_tokens, num_new_tokens
        )
        if num_new_tokens > 0:
            self._attn_work += num_new_tokens * (
                num_computed_tokens + num_new_tokens / 2
            )
            self._attn_prefill_scheduled |= prefill
        return num_new_tokens

    @staticmethod
    def _remaining(request: Request) -> int:
        return request.num_tokens - request.num_computed_tokens

    def _is_stalled(self, request: Request) -> bool:
        return (
            self._stalled_steps.get(request.request_id, 0) >= self.max_consecutive_holds
        )

    def _order_running(self) -> None:
        decoding = [r for r in self.running if not r.is_prefill_chunk]
        prefilling = sorted(
            (r for r in self.running if r.is_prefill_chunk),
            key=lambda r: (not self._is_stalled(r), self._remaining(r)),
        )
        self.running[:] = decoding + prefilling

    def _prune_hold_state(self) -> None:
        prefilling = {r.request_id for r in self.running if r.is_prefill_chunk}
        self._stalled_steps = {
            k: v for k, v in self._stalled_steps.items() if k in prefilling
        }
        if self._hold_backoff:
            waiting = {
                r.request_id for r in chain(self.kv_holding_waiting, self.waiting)
            }
            self._hold_backoff = {
                k: v for k, v in self._hold_backoff.items() if k in waiting
            }

    def _record_hold_outcome(self, target: Request) -> None:
        # Admission either runs the target or starts its KV load; both move it
        # out of WAITING / PREEMPTED.
        if target.status not in (RequestStatus.WAITING, RequestStatus.PREEMPTED):
            self._hold_backoff.pop(target.request_id, None)
            return
        failures = self._hold_backoff.get(target.request_id, (0, 0))[0] + 1
        delay = 1 << min(failures - 1, _MAX_BACKOFF_SHIFT)
        self._hold_backoff[target.request_id] = (
            failures,
            self.current_step + 1 + delay,
        )

    def _long_prefill_threshold(self, num_eligible_reqs: int) -> int:
        # Mirrors the long_prefill_token_threshold computation inline in
        # Scheduler.schedule() (vLLM v0.33.0, scheduler.py ~L754-769). Re-sync
        # on vLLM upgrades.
        lptt = (
            self.scheduler_config.long_prefill_token_threshold
            if num_eligible_reqs > 1
            else 0
        )
        if lptt > 0 and self.adaptive_long_prefill_threshold:
            lptt = max(
                lptt,
                self.scheduler_config.max_num_batched_tokens // num_eligible_reqs,
            )
        return lptt

    def _plan_running(
        self, reqs: list[Request], lptt: int
    ) -> tuple[dict[str, int], int, float, bool]:
        """Tokens the RUNNING pass of Scheduler.schedule() would give ``reqs``,
        the budget it leaves for WAITING, and (with the attention work cap)
        the attention work it uses and whether it schedules a prefill chunk.

        Mirrors the per-request ``num_new_tokens`` math inline in that pass
        (vLLM v0.33.0, scheduler.py ~L793-887); re-sync on vLLM upgrades.
        Encoder, Mamba-alignment and KV-allocation limits are not modelled, so
        a misestimate only makes a hold miss, which the backoff absorbs.
        """
        next_step = self.current_step + 1
        spec = self.vllm_config.speculative_config
        draft_slots = spec.max_num_new_slots_for_drafting if spec is not None else 0
        token_budget = self.max_num_scheduled_tokens
        input_budget = self.scheduler_config.max_num_batched_tokens
        alloc: dict[str, int] = {}
        work = 0.0
        prefill_scheduled = False
        for r in reqs:
            if token_budget <= 0 or input_budget <= draft_slots:
                break
            if (
                r.num_output_placeholders > 0
                and r.num_computed_tokens + 2 - r.num_output_placeholders
                >= r.num_prompt_tokens + r.max_tokens
            ) or next_step < r.next_decode_eligible_step:
                continue
            n = (
                r.num_tokens_with_spec
                + r.num_output_placeholders
                - r.num_computed_tokens
            )
            if 0 < lptt < n:
                n = lptt
            n = min(
                n,
                token_budget,
                input_budget - draft_slots,
                self.max_model_len
                - r.num_computed_tokens
                - self.num_sampled_tokens_per_step,
            )
            prefill = r.num_computed_tokens < r.num_prompt_tokens
            if self._attn_cap_on and prefill and n > 0:
                n = self._attn_cap_tokens(
                    r.num_computed_tokens, n, work, prefill_scheduled
                )
            n = self._reserve_prefill_lookahead(r, r.num_computed_tokens, n)
            if n <= 0:
                continue
            alloc[r.request_id] = n
            token_budget -= n
            input_budget -= n + draft_slots
            work += n * (r.num_computed_tokens + n / 2)
            prefill_scheduled |= prefill
        return (
            alloc,
            min(token_budget, input_budget - draft_slots),
            work,
            prefill_scheduled,
        )

    def _next_waiting(self) -> Request | None:
        """The request upstream's WAITING pass admits next."""
        return next(
            (
                r
                for r in chain(self.kv_holding_waiting, self.waiting)
                if not self._is_blocked_waiting_status(r.status)
            ),
            None,
        )

    def _prefix_hit(self, request: Request) -> tuple[KVCacheBlocks, int]:
        if request.num_computed_tokens > 0:
            return (
                self.kv_cache_manager.empty_kv_cache_blocks,
                request.num_computed_tokens,
            )
        blocks, num_hit_tokens, _, _ = self._get_local_prefix_cache_hit(request)
        return blocks, num_hit_tokens

    def _select_chunks_to_hold(self) -> tuple[list[Request], Request | None]:
        """Return ``(chunks_to_hold, target_waiting_request)``."""
        if not any(r.is_prefill_chunk for r in self.running):
            return [], None
        if (
            len(self.running) + self.num_waiting_for_streaming_input
            >= self.max_num_active_reqs
        ):
            return [], None
        target = self._next_waiting()
        if target is None:
            return [], None
        _, retry_step = self._hold_backoff.get(target.request_id, (0, 0))
        if self.current_step + 1 < retry_step:
            return [], None

        hit_blocks, hit_tokens = self._prefix_hit(target)
        est_hit_tokens = self._estimated_prefix_hit_tokens(target, hit_tokens)
        target_remaining = max(target.num_tokens - est_hit_tokens, 1)
        _, num_waiting = self.get_request_counts()
        kept = list(self.running)
        held: list[Request] = []
        while True:
            lptt = self._long_prefill_threshold(len(kept) + num_waiting)
            alloc, left, work, prefill_scheduled = self._plan_running(kept, lptt)
            need = min(target_remaining, lptt) if lptt > 0 else target_remaining
            admit = min(need, left)
            if self._attn_cap_on and admit > 0:
                admit = self._attn_cap_tokens(
                    est_hit_tokens, admit, work, prefill_scheduled
                )
            # Enough budget, and the cap lets the target reach the next tile.
            if left >= need and admit >= min(
                need, self._tokens_to_next_tile(est_hit_tokens)
            ):
                break
            # Non-stalled prefill chunks sit last in `kept`, ordered by
            # remaining tokens; when `left >= need`, the cap is what shrank
            # `admit`, so hold the candidate that frees the most attention work.
            candidates = [
                r
                for r in reversed(kept)
                if r.request_id in alloc
                and r.is_prefill_chunk
                and self._remaining(r) > target_remaining
                and not self._is_stalled(r)
            ]
            if not candidates:
                break
            candidate = (
                max(
                    candidates,
                    key=lambda r: alloc[r.request_id]
                    * (r.num_computed_tokens + alloc[r.request_id] / 2),
                )
                if left >= need
                else candidates[0]
            )
            kept.remove(candidate)
            held.append(candidate)
        if not held or admit <= 0:
            return [], None
        kv_admit_tokens = (est_hit_tokens - hit_tokens) + admit
        if not self._kv_fits_hold(
            kept, alloc, target, hit_blocks, hit_tokens, kv_admit_tokens
        ):
            return [], None
        return held, target

    def _kv_fits_hold(
        self,
        kept: list[Request],
        alloc: dict[str, int],
        target: Request,
        hit_blocks: KVCacheBlocks,
        hit_tokens: int,
        target_tokens: int,
    ) -> bool:
        """Whether free KV blocks cover ``kept`` running requests plus
        ``target``, counted as KVCacheManager.allocate_slots() counts them,
        so a hold never forces a preemption."""
        coordinator = self.kv_cache_manager.coordinator
        empty_blocks = self.kv_cache_manager.empty_kv_cache_blocks.blocks

        def blocks_for(
            request: Request,
            computed_tokens: int,
            num_new_tokens: int,
            computed_blocks: tuple,
        ) -> int:
            num_tokens_main = computed_tokens + num_new_tokens
            return coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=min(
                    num_tokens_main + self.num_lookahead_tokens, self.max_model_len
                ),
                new_computed_blocks=computed_blocks,
                num_encoder_tokens=0,
                total_computed_tokens=computed_tokens,
                num_local_computed_tokens=computed_tokens,
                num_tokens_main_model=num_tokens_main,
                prefill_end=max(request.num_prompt_tokens, request.num_tokens - 1),
            )

        needed = sum(
            blocks_for(r, r.num_computed_tokens, alloc[r.request_id], empty_blocks)
            for r in kept
            # In-flight prefills are covered by the reservation below.
            if r.request_id in alloc and r not in self._inflight_prefills
        )
        target_blocks = blocks_for(target, hit_tokens, target_tokens, hit_blocks.blocks)
        if self.scheduler_reserve_full_isl:
            full_num_tokens = min(target.num_tokens, self.max_model_len)
            target_blocks = max(
                target_blocks,
                coordinator.get_num_blocks_to_allocate(
                    request_id=target.request_id,
                    num_tokens=full_num_tokens,
                    new_computed_blocks=hit_blocks.blocks,
                    num_encoder_tokens=0,
                    total_computed_tokens=hit_tokens,
                    num_local_computed_tokens=hit_tokens,
                    num_tokens_main_model=full_num_tokens,
                    apply_admission_cap=True,
                    prefill_end=max(target.num_prompt_tokens, target.num_tokens - 1),
                ),
            )
        needed += target_blocks
        if kept:
            # allocate_slots() adds the watermark when others are scheduled.
            needed += self.kv_cache_manager.watermark_blocks
        reserved = (
            self._inflight_prefill_reserved_blocks() + self._spec_decode_step_blocks
        )
        free = self.kv_cache_manager.block_pool.get_num_free_blocks()
        return free - reserved >= needed
