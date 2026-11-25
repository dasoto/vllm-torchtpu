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
from collections import OrderedDict, deque
from itertools import chain, islice
from typing import Any, cast

from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
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
_WARM_HOLD_HEAD_SECONDS = 90.0
_CRITICAL_HOLD_SECONDS = 240.0
_WARM_FOLLOWUP_MIN_HIT_TOKENS = 8192
_WARM_FOLLOWUP_MAX_UNCACHED_TOKENS = 32768
_MAX_EXTERNAL_BLOCK_HASHES = 524288
_MAX_PRESTAGED_KV_HOLDING = 2
_MAX_RECENT_PROMPT_PREFIXES = 128
_MAX_PINNED_JUNCTIONS = 4
_MIN_SHARED_PREFIX_TOKENS = 1024


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
        self.session_priority = envs.VLLM_TPU_SRPT_SESSION_PRIORITY
        self.max_prestaged_kv_holding = max(
            int(envs.VLLM_TPU_SRPT_MAX_PRESTAGED_KV_HOLDING), 1
        )
        coordinator = getattr(self.kv_cache_manager, "coordinator", None)
        self.shared_prefix_checkpoint_enabled = bool(
            envs.VLLM_TPU_SHARED_PREFIX_CHECKPOINT
            and getattr(self, "mamba_partial_cache_hit", False)
            and coordinator is not None
            and getattr(coordinator, "num_reprefillable_tokens", 0) == 0
        )
        if self.shared_prefix_checkpoint_enabled and coordinator is not None:
            self.mamba_shared_prefix_checkpoint = True
            for mgr in coordinator.single_type_managers:
                if hasattr(mgr, "shared_prefix_checkpoint"):
                    mgr.shared_prefix_checkpoint = True
        self._recent_prompt_prefixes: OrderedDict[str, tuple[Any, ...]] = OrderedDict()
        self._inflight_junctions: dict[str, tuple[int, Any]] = {}
        self._ready_fa_junctions: dict[str, tuple[int, Any, list[tuple[Any, Any]]]] = {}
        self._deferred_fa_junction_pubs: deque[
            tuple[int, Request, list[tuple[Any, Any]], int, Any]
        ] = deque()
        self._pinned_junction_hashes: set[Any] = set()
        self._pinned_junction_blocks: list[tuple[Any, Any]] = []
        # Whether another DP rank decoded in the last step the runner reported,
        # and whether the cap applies to the step being scheduled.
        self._other_rank_decoding = False
        self._attn_cap_on = False
        # Attention work and whether a prefill chunk was scheduled so far in
        # this step's Scheduler.schedule(), and the requests the cap shrank.
        self._in_schedule = False
        self._cross_rank_decode_hold_active = False
        self._running_chunk_caps: dict[str, int] = {}
        self._attn_work = 0.0
        self._attn_prefill_scheduled = False
        self._attn_capped: set[str] = set()
        self._attn_steps = 0
        self._attn_cap_on_steps = 0
        self._attn_capped_chunks = 0
        logger.info(
            "SrptPrefillScheduler active (max_consecutive_holds=%d, "
            "attn_work_cap=%d, attn_tile_tokens=%d, cadence_no_escape=%s, "
            "promote_warm_waiting=%s, session_priority=%s, "
            "shared_prefix_checkpoint=%s)",
            self.max_consecutive_holds,
            self.attn_work_cap,
            self.attn_tile_tokens,
            self.cadence_no_escape,
            self.promote_warm_waiting,
            self.session_priority,
            self.shared_prefix_checkpoint_enabled,
        )

    def _record_external_block_hashes(self, output: SchedulerOutput) -> None:
        if self.connector is None or not output.num_scheduled_tokens:
            return
        tokens_per_hash = max(
            int(getattr(self, "hash_block_size", 0) or self.cache_config.block_size or 1),
            1,
        )
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
        tokens_per_hash = max(
            int(getattr(self, "hash_block_size", 0) or self.cache_config.block_size or 1),
            1,
        )
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
        check_full_isl = self.scheduler_reserve_full_isl or (
            self.connector is not None
            and self.promote_warm_waiting
            and (is_async_ext or reserved_async_blocks > 0)
        )
        if not check_full_isl and not is_async_ext and self.connector is None:
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
        effective_free = (
            free_blocks - reserved_async_blocks
            if (is_async_ext or self.promote_warm_waiting or self.connector is not None)
            else free_blocks
        )
        if check_full_isl:
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
            if full_needed + watermark > effective_free:
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
        elif not check_full_isl:
            rem = max(request.num_tokens - total_comp, 1)
            lptt = self.scheduler_config.long_prefill_token_threshold
            step_new = min(rem, lptt) if lptt > 0 else rem
            step_new = min(step_new, self.scheduler_config.max_num_batched_tokens)
            num_main = min(total_comp + step_new, self.max_model_len)
            step_needed = self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=min(
                    num_main + self.num_lookahead_tokens, self.max_model_len
                ),
                new_computed_blocks=computed_blocks,
                num_encoder_tokens=0,
                total_computed_tokens=total_comp,
                num_local_computed_tokens=num_local,
                num_tokens_main_model=num_main,
                prefill_end=prefill_end,
            )
            if step_needed + watermark > effective_free:
                return False, local_hit, est_hit
        return True, local_hit, est_hit

    def _release_pinned_junction_blocks(self) -> None:
        for _, (_, _, mamba_pins) in list(self._ready_fa_junctions.items()):
            for pool, blk in mamba_pins:
                if not blk.is_null and blk.ref_cnt > 0:
                    pool.free_blocks([blk])
        self._ready_fa_junctions.clear()
        self._inflight_junctions.clear()
        self._deferred_fa_junction_pubs.clear()
        for pool, blk in self._pinned_junction_blocks:
            if not blk.is_null and blk.ref_cnt > 0:
                pool.free_blocks([blk])
        self._pinned_junction_blocks.clear()
        self._pinned_junction_hashes.clear()
        self._recent_prompt_prefixes.clear()

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        self._release_pinned_junction_blocks()
        ok = super().reset_prefix_cache(
            reset_running_requests=reset_running_requests,
            reset_connector=reset_connector,
        )
        if ok and reset_connector:
            self._external_block_hashes.clear()
        return ok

    def _detect_shared_prefix_boundary(
        self,
        request: Request,
        num_local: int,
        max_cache_hit_length: int | None = None,
    ) -> int:
        if (
            not self.shared_prefix_checkpoint_enabled
            or len(self._pinned_junction_hashes) >= _MAX_PINNED_JUNCTIONS
            or request.skip_reading_prefix_cache
            or not request.block_hashes
        ):
            return 0
        hash_bs = int(getattr(self, "hash_block_size", 0) or 0)
        super_bs = int(self.block_size or 0)
        if hash_bs <= 0 or super_bs <= hash_bs or num_local % super_bs != 0:
            return 0
        max_hashes = min(len(request.block_hashes), 2 * (super_bs // hash_bs))
        min_hashes = max(_MIN_SHARED_PREFIX_TOKENS // hash_bs, 1)
        if max_hashes < min_hashes:
            return 0
        prefix = tuple(request.block_hashes[:max_hashes])
        best_lcp = 0
        for other_id, other_prefix in self._recent_prompt_prefixes.items():
            if other_id == request.request_id:
                continue
            limit = min(len(prefix), len(other_prefix))
            if limit <= best_lcp:
                continue
            lcp = 0
            while lcp < limit and prefix[lcp] == other_prefix[lcp]:
                lcp += 1
            if lcp > best_lcp:
                best_lcp = lcp

        if request.request_id in self._recent_prompt_prefixes:
            self._recent_prompt_prefixes.move_to_end(request.request_id)
        else:
            if len(self._recent_prompt_prefixes) >= _MAX_RECENT_PROMPT_PREFIXES:
                self._recent_prompt_prefixes.popitem(last=False)
        self._recent_prompt_prefixes[request.request_id] = prefix

        if best_lcp < min_hashes:
            return 0
        lcp_tokens = best_lcp * hash_bs
        if max_cache_hit_length is not None and lcp_tokens > max_cache_hit_length:
            lcp_tokens = (max_cache_hit_length // hash_bs) * hash_bs
        if (
            lcp_tokens <= num_local
            or lcp_tokens < _MIN_SHARED_PREFIX_TOKENS
            or lcp_tokens >= request.num_prompt_tokens
            or lcp_tokens % super_bs == 0
        ):
            return 0
        junction_hash = prefix[lcp_tokens // hash_bs - 1]
        if junction_hash in self._pinned_junction_hashes:
            return 0
        active_hashes = (
            {h for _, h in self._inflight_junctions.values()}
            | {h for _, h, _ in self._ready_fa_junctions.values()}
            | {h for _, _, _, _, h in self._deferred_fa_junction_pubs}
        )
        if junction_hash in active_hashes:
            return 0
        self._inflight_junctions[request.request_id] = (lcp_tokens, junction_hash)
        return lcp_tokens

    def _get_local_prefix_cache_hit(
        self, request: Request, max_cache_hit_length: int | None = None
    ) -> tuple[KVCacheBlocks, int, int, bool]:
        blocks, num_local, shared_prefix_boundary, hit_diverged = (
            super()._get_local_prefix_cache_hit(request, max_cache_hit_length)
        )
        if (
            self.shared_prefix_checkpoint_enabled
            and self._in_schedule
            and shared_prefix_boundary == 0
            and request.num_computed_tokens == 0
        ):
            shared_prefix_boundary = self._detect_shared_prefix_boundary(
                request, num_local, max_cache_hit_length
            )
        return blocks, num_local, shared_prefix_boundary, hit_diverged

    def _stage_ready_fa_junctions(self, prev_sched_seq: int) -> None:
        if not self._ready_fa_junctions:
            return
        coordinator = self.kv_cache_manager.coordinator
        fa_mgrs = [
            mgr
            for mgr in coordinator.single_type_managers
            if isinstance(mgr, FullAttentionManager)
            and mgr.kv_cache_spec.prefix_cacheable
        ]
        for req_id, (j_tokens, j_hash, mamba_pins) in list(
            self._ready_fa_junctions.items()
        ):
            req = self.requests.get(req_id)
            if (
                req is None
                or req.status != RequestStatus.RUNNING
                or req.num_computed_tokens != j_tokens
            ):
                for pool, blk in mamba_pins:
                    if not blk.is_null and blk.ref_cnt > 0:
                        pool.free_blocks([blk])
                self._ready_fa_junctions.pop(req_id, None)
                continue
            free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
            watermark = self.kv_cache_manager.watermark_blocks
            if free_blocks < watermark + len(fa_mgrs) + 4:
                continue
            staged_fa: list[tuple[Any, Any]] = []
            valid = True
            for fa_mgr in fa_mgrs:
                b_idx = j_tokens // fa_mgr.block_size
                req_blocks = fa_mgr.req_to_blocks.get(req_id, ())
                if b_idx >= len(req_blocks) or req_blocks[b_idx].is_null:
                    valid = False
                    break
            if not valid:
                for pool, blk in mamba_pins:
                    if not blk.is_null and blk.ref_cnt > 0:
                        pool.free_blocks([blk])
                req.shared_prefix_boundary = 0
                self._ready_fa_junctions.pop(req_id, None)
                continue
            for fa_mgr in fa_mgrs:
                b_idx = j_tokens // fa_mgr.block_size
                source_block = fa_mgr.req_to_blocks[req_id][b_idx]
                cow_block = fa_mgr.block_pool.get_new_blocks(1)[0]
                if fa_mgr.records_new_block_ids:
                    fa_mgr.new_block_ids.append(cow_block.block_id)
                # take_kv_cache_block_copies() retains both endpoints and
                # releases 1 ref from each at `prev_sched_seq + 1`. Add 1 ref
                # to `source_block` so `req` keeps its ref, and 1 extra ref to
                # `cow_block` so `cow_block` stays pinned (ref_cnt == 1) in HBM.
                source_block.ref_cnt += 1
                cow_block.ref_cnt += 1
                fa_mgr._pending_cow_copies.append((source_block, cow_block))
                self._pinned_junction_blocks.append((fa_mgr.block_pool, cow_block))
                staged_fa.append((fa_mgr, cow_block))
            self._pinned_junction_blocks.extend(mamba_pins)
            req.shared_prefix_boundary = 0
            self._ready_fa_junctions.pop(req_id, None)
            self._deferred_fa_junction_pubs.append(
                (prev_sched_seq + 1, req, staged_fa, j_tokens, j_hash)
            )

    def _capture_inflight_junctions(self, output: SchedulerOutput) -> None:
        if not self._inflight_junctions:
            return
        coordinator = self.kv_cache_manager.coordinator
        for req_id, (j_tokens, j_hash) in list(self._inflight_junctions.items()):
            req = self.requests.get(req_id)
            if (
                req is None
                or req.status != RequestStatus.RUNNING
                or req.num_computed_tokens > j_tokens
            ):
                self._inflight_junctions.pop(req_id, None)
                continue
            if (
                req.num_computed_tokens == j_tokens
                and output.num_scheduled_tokens.get(req_id, 0) > 0
            ):
                mamba_pins: list[tuple[Any, Any]] = []
                for mgr in coordinator.single_type_managers:
                    if (
                        getattr(mgr, "mamba_cache_mode", None) == "align"
                        and j_tokens % mgr.block_size == 0
                    ):
                        m_idx = j_tokens // mgr.block_size - 1
                        m_blocks = mgr.req_to_blocks.get(req_id, ())
                        if (
                            0 <= m_idx < len(m_blocks)
                            and not m_blocks[m_idx].is_null
                            and m_blocks[m_idx].block_hash is not None
                        ):
                            mgr.block_pool.touch([m_blocks[m_idx]])
                            mamba_pins.append((mgr.block_pool, m_blocks[m_idx]))
                self._inflight_junctions.pop(req_id, None)
                self._ready_fa_junctions[req_id] = (j_tokens, j_hash, mamba_pins)

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
            is_warm_session = est_hit >= _WARM_FOLLOWUP_MIN_HIT_TOKENS
            if self.session_priority:
                is_warm_or_preempted = (
                    req.num_preemptions > 0 or uncached < 8192 or is_warm_session
                )
                hold_head_s = (
                    0.0
                    if req.num_preemptions > 0
                    else (_WARM_HOLD_HEAD_SECONDS if is_warm_or_preempted else 900.0)
                )
                can_drain_head = (
                    req.num_preemptions > 0
                    or (is_warm_session and wait_s > _HARD_STARVATION_SECONDS)
                    or (not has_inflight_prefills and wait_s > hold_head_s)
                )
                exceeds_hbm = 0 if (fits or can_drain_head) else 1
                if (
                    req.num_preemptions > 0
                    or (is_warm_session and wait_s > _HARD_STARVATION_SECONDS)
                    or (not fits and can_drain_head)
                ):
                    tier = -1
                    primary_metric = req.arrival_time
                elif uncached < 8192 or (
                    is_warm_session and wait_s > _WARM_STARVATION_SECONDS
                ):
                    tier = 0
                    primary_metric = float(uncached)
                elif is_warm_session or uncached < 32768:
                    tier = 1
                    primary_metric = float(uncached)
                elif uncached < 131072:
                    tier = 2
                    primary_metric = req.arrival_time
                else:
                    tier = 3
                    primary_metric = req.arrival_time
            else:
                is_warm_or_preempted = (
                    req.num_preemptions > 0
                    or uncached <= 8192
                    or (
                        is_warm_session
                        and uncached <= _WARM_FOLLOWUP_MAX_UNCACHED_TOKENS
                    )
                    or wait_s > _WARM_STARVATION_SECONDS
                )
                # Upstream Scheduler.schedule() executes `break` (head-of-line block)
                # when waiting[0] fails KV allocation. Demote freshly arrived
                # non-preempted requests that cannot currently pass allocate_slots
                # behind requests that CAN fit, while letting preempted streams
                # immediately (0s) and aged warm/cold requests (15s / 60s) hold the
                # head in tier -1 so HBM blocks drain for deep multi-turn sessions
                # instead of starving behind a continuous stream of smaller arrivals.
                hold_head_s = (
                    0.0
                    if req.num_preemptions > 0
                    else (
                        _WARM_HOLD_HEAD_SECONDS
                        if is_warm_or_preempted
                        else _CRITICAL_HOLD_SECONDS
                    )
                )
                # Holding waiting[0] while `_inflight_prefills` is non-empty cannot
                # stop running prefill chunks from allocating blocks and only
                # head-of-line blocks fitting warm follow-ups behind `_inflight_prefill_reserved_blocks()`.
                can_drain_head = (
                    req.num_preemptions > 0
                    or wait_s > _HARD_STARVATION_SECONDS
                    or (not has_inflight_prefills and wait_s > hold_head_s)
                )
                exceeds_hbm = 0 if (fits or can_drain_head) else 1
                if (
                    req.num_preemptions > 0
                    or wait_s > _HARD_STARVATION_SECONDS
                    or (not fits and can_drain_head)
                ):
                    tier = -1
                elif uncached <= 8192 or wait_s > _WARM_STARVATION_SECONDS:
                    tier = 0
                elif (
                    is_warm_session and uncached <= _WARM_FOLLOWUP_MAX_UNCACHED_TOKENS
                ):
                    tier = 1
                else:
                    tier = 2
                primary_metric = req.arrival_time if tier == -1 else float(uncached)
            demote_stalled_fetch = (
                1 if (is_stalled_fetch and not has_inflight_prefills) else 0
            )
            # On decode-only cadence steps (throttle_prefills=True), non-external
            # prefills hit `break` at defer_prefills whereas external async loads
            # (`load_kv_async=True`, 0 compute tokens) can launch Host->HBM DMA.
            not_async_ext = 0 if (throttle_prefills and est_hit > local_hit) else 1
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

    def _is_ready_waiting_request(self, request: Request) -> bool:
        if not self._is_blocked_waiting_status(request.status):
            return True
        return (
            request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
            and request.request_id in self.finished_recving_kv_req_ids
        )

    def _kv_holding_req_fits(self, request: Request, free_blocks: int) -> bool:
        """Whether a completed-load request in ``kv_holding_waiting`` passes ``allocate_slots``."""
        comp = min(request.num_computed_tokens, self.max_model_len)
        if comp <= 0:
            return True
        empty_blocks = self.kv_cache_manager.empty_kv_cache_blocks.blocks
        watermark = self.kv_cache_manager.watermark_blocks if self.running else 0
        prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
        coordinator = self.kv_cache_manager.coordinator
        if self.scheduler_reserve_full_isl:
            full_num_tokens = min(request.num_tokens, self.max_model_len)
            needed = coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=full_num_tokens,
                new_computed_blocks=empty_blocks,
                num_encoder_tokens=0,
                total_computed_tokens=comp,
                num_local_computed_tokens=comp,
                num_tokens_main_model=full_num_tokens,
                apply_admission_cap=True,
                prefill_end=prefill_end,
            )
            if needed + watermark > free_blocks:
                return False
        rem = max(request.num_tokens - comp, 1)
        lptt = self.scheduler_config.long_prefill_token_threshold
        step_new = min(rem, lptt) if lptt > 0 else rem
        step_new = min(step_new, self.scheduler_config.max_num_batched_tokens)
        num_main = comp + step_new
        needed_step = coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=min(num_main + self.num_lookahead_tokens, self.max_model_len),
            new_computed_blocks=empty_blocks,
            num_encoder_tokens=0,
            total_computed_tokens=comp,
            num_local_computed_tokens=comp,
            num_tokens_main_model=num_main,
            prefill_end=prefill_end,
        )
        return needed_step + watermark <= free_blocks

    def _maybe_order_kv_holding_waiting(self) -> None:
        """Order ready ``kv_holding_waiting`` entries so fitting ones drain first."""
        if self.policy != SchedulingPolicy.FCFS or len(self.kv_holding_waiting) < 2:
            return
        items = list(self.kv_holding_waiting)
        ready = [
            r
            for r in items
            if self._is_ready_waiting_request(r) and r.num_computed_tokens > 0
        ]
        if len(ready) < 2:
            return
        free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
        ordered_ready = sorted(
            ready,
            key=lambda r: (
                0 if self._kv_holding_req_fits(r, free_blocks) else 1,
                self._remaining(r),
                r.arrival_time,
            ),
        )
        if [r.request_id for r in ready] == [r.request_id for r in ordered_ready]:
            return
        ready_iter = iter(ordered_ready)
        ready_ids = {r.request_id for r in ready}
        rebuilt = [
            next(ready_iter) if r.request_id in ready_ids else r for r in items
        ]
        while self.kv_holding_waiting:
            self.kv_holding_waiting.pop_request()
        for req_r in reversed(rebuilt):
            self.kv_holding_waiting.prepend_request(req_r)

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        local_decoding = any(not r.is_prefill_chunk for r in self.running)
        any_decoding = local_decoding or self._other_rank_decoding
        if self.cadence_no_escape and not any_decoding:
            throttle_prefills = False
        # The cap only protects decodes: it applies while this rank decodes in
        # this step or another rank decoded in the last reported step.
        self._attn_cap_on = bool(self.attn_work_cap) and any_decoding
        if self.cadence_no_escape and any_decoding:
            self.prefill_capacity_bound = False
        self._maybe_promote_warm_waiting(throttle_prefills=throttle_prefills)

        held: list[Request] = []
        stashed_kv_holding: list[Request] = []
        stashed_excess_async: list[Request] = []
        original_waiting_order: list[Request] = []
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
        self._cross_rank_decode_hold_active = False
        self._running_chunk_caps.clear()
        if cross_rank_decode_hold:
            held = [r for r in self.running if r.is_prefill_chunk]
            if self.connector is None:
                if prev_pause_state == PauseState.UNPAUSED:
                    self._pause_state = PauseState.PAUSED_NEW
            else:
                self._cross_rank_decode_hold_active = True
            self.max_num_scheduled_tokens = min(
                prev_max_scheduled, self.scheduler_config.max_num_seqs
            )
        elif not throttle_prefills and self.policy == SchedulingPolicy.FCFS:
            self._maybe_order_kv_holding_waiting()
            self._prune_hold_state()
            self._order_running()
            prefilling = [r for r in self.running if r.is_prefill_chunk]
            held, target = self._select_chunks_to_hold()
            if (
                self.connector is not None
                and self.promote_warm_waiting
                and self.kv_holding_waiting
            ):
                # Temporarily step past already-polled kv_holding_waiting entries
                # whose tail blocks cannot currently fit in free HBM so upstream
                # allocate_slots() does not return None -> `break` and block
                # smaller fitting warm/short requests behind them.
                free_kh = self.kv_cache_manager.block_pool.get_num_free_blocks()
                now_kh = time.time()
                keep_kh_ready: list[Request] = []
                while self.kv_holding_waiting:
                    req_kh = self.kv_holding_waiting.pop_request()
                    if (
                        req_kh.status == RequestStatus.WAITING
                        and 0 < req_kh.num_computed_tokens < req_kh.num_tokens - 1
                        and not self._kv_holding_req_fits(req_kh, free_kh)
                        and (now_kh - req_kh.arrival_time) <= _HARD_STARVATION_SECONDS
                    ):
                        stashed_kv_holding.append(req_kh)
                    else:
                        keep_kh_ready.append(req_kh)
                for req_keep_r in reversed(keep_kh_ready):
                    self.kv_holding_waiting.prepend_request(req_keep_r)

        if self.connector is not None and self.waiting:
            # Bound concurrent pre-staged external async loads in kv_holding_waiting
            # (`_MAX_PRESTAGED_KV_HOLDING`) so a burst of 0-compute async loads
            # inside one step or decode cycle cannot hoard HBM KV blocks needed
            # by already-fetched warm requests and active decodes. On
            # `cross_rank_decode_hold` steps, also stash non-async prefill
            # requests so `self.waiting` only initiates 0-compute async KV loads.
            allowed_new_async = max(
                0, self.max_prestaged_kv_holding - len(self.kv_holding_waiting)
            )
            seen_async = 0
            kept_waiting: list[Request] = []
            original_waiting_order = list(self.waiting)
            for req_w in original_waiting_order:
                if not self._is_blocked_waiting_status(req_w.status):
                    _, loc_h = self._prefix_hit(req_w)
                    is_async_candidate = (
                        req_w.num_computed_tokens == 0
                        and self._estimated_prefix_hit_tokens(req_w, loc_h) > loc_h
                    )
                    if is_async_candidate:
                        if seen_async >= allowed_new_async:
                            stashed_excess_async.append(req_w)
                            continue
                        seen_async += 1
                    elif (
                        cross_rank_decode_hold
                        and req_w.num_computed_tokens < req_w.num_tokens - 1
                    ):
                        stashed_excess_async.append(req_w)
                        continue
                kept_waiting.append(req_w)
            if stashed_excess_async:
                while self.waiting:
                    self.waiting.pop_request()
                for req_k in reversed(kept_waiting):
                    self.waiting.prepend_request(req_k)

        if (
            throttle_prefills
            and self.connector is not None
            and self.kv_holding_waiting
        ):
            # Completed async KV loads waiting for a prefill step sit in
            # kv_holding_waiting and trigger `break` at `defer_prefills` (or
            # would run prefill tokens under `cross_rank_decode_hold`),
            # preventing `self.waiting` from starting new 0-compute async DMA
            # loads during decode steps. Temporarily step past them while
            # keeping WAITING_FOR_REMOTE_KVS entries visible for poll completion.
            total_prestaged = len(self.kv_holding_waiting)
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
            for req_keep in reversed(keep_kv_holding):
                self.kv_holding_waiting.prepend_request(req_keep)
            if (
                total_prestaged >= self.max_prestaged_kv_holding
                and prev_pause_state == PauseState.UNPAUSED
            ):
                self._pause_state = PauseState.PAUSED_NEW

        held_ids = {r.request_id for r in held}
        self.running[:] = [r for r in self.running if r.request_id not in held_ids]
        self.max_num_running_reqs -= len(held)
        self.max_num_active_reqs -= len(held)
        self._attn_work = 0.0
        self._attn_prefill_scheduled = False
        self._attn_capped.clear()
        prev_sched_seq = getattr(self, "sched_step_seq", 0)
        if self.shared_prefix_checkpoint_enabled and not cross_rank_decode_hold:
            self._stage_ready_fa_junctions(prev_sched_seq)
        capped_by_share = set(self._running_chunk_caps)
        self._in_schedule = True
        try:
            output = super().schedule(throttle_prefills)
            if self.cadence_no_escape and any_decoding:
                self.prefill_capacity_bound = False
        finally:
            self._in_schedule = False
            self._cross_rank_decode_hold_active = False
            self._running_chunk_caps.clear()
            self._pause_state = prev_pause_state
            self.max_num_scheduled_tokens = prev_max_scheduled
            self.max_num_running_reqs += len(held)
            self.max_num_active_reqs += len(held)
            self.running.extend(held)
            for req_stashed in reversed(stashed_kv_holding):
                self.kv_holding_waiting.prepend_request(req_stashed)
            if stashed_excess_async:
                remaining_waiting = list(self.waiting)
                remaining_ids = {r.request_id for r in remaining_waiting}
                stashed_ids = {r.request_id for r in stashed_excess_async}
                restored = [
                    r
                    for r in original_waiting_order
                    if r.request_id in remaining_ids or r.request_id in stashed_ids
                ]
                known_ids = {r.request_id for r in restored}
                restored.extend(
                    r for r in remaining_waiting if r.request_id not in known_ids
                )
                while self.waiting:
                    self.waiting.pop_request()
                for req_rest in reversed(restored):
                    self.waiting.prepend_request(req_rest)

        if self.shared_prefix_checkpoint_enabled:
            self._capture_inflight_junctions(output)
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
                # Got all the attention work cap or budget-sharing cap allowed.
                or (
                    n > 0
                    and (
                        r.request_id in self._attn_capped
                        or r.request_id in capped_by_share
                    )
                )
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
        res = super().update_from_output(scheduler_output, model_runner_output)
        if self._deferred_fa_junction_pubs:
            proc_seq = getattr(self, "processed_step_seq", 0)
            defer_free = bool(getattr(self, "defer_block_free", False))
            completed_dst_ids = {
                c.dst_block_id for c in (scheduler_output.kv_cache_block_copies or ())
            }
            remaining_pubs: deque[
                tuple[int, Request, list[tuple[Any, Any]], int, Any]
            ] = deque()
            for entry in self._deferred_fa_junction_pubs:
                fence_seq, req_pub, staged_fa, j_tokens, j_hash = entry
                retired = (defer_free and proc_seq >= fence_seq) or (
                    bool(completed_dst_ids)
                    and all(
                        cow_block.block_id in completed_dst_ids
                        for _, cow_block in staged_fa
                    )
                )
                if not retired:
                    remaining_pubs.append(entry)
                    continue
                for fa_mgr, cow_block in staged_fa:
                    if not cow_block.is_null and cow_block.ref_cnt > 0:
                        fa_mgr.block_pool.cache_partial_block(
                            request=req_pub,
                            block=cow_block,
                            num_tokens=j_tokens,
                            kv_cache_group_id=fa_mgr.kv_cache_group_id,
                            block_size=fa_mgr.block_size,
                        )
                self._pinned_junction_hashes.add(j_hash)
                logger.info(
                    "SrptPrefillScheduler: pinned shared prefix junction at %d "
                    "tokens in HBM (%d pinned junctions).",
                    j_tokens,
                    len(self._pinned_junction_hashes),
                )
            self._deferred_fa_junction_pubs = remaining_pubs
        return res

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

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        if self._in_schedule and self._running_chunk_caps:
            cap = self._running_chunk_caps.get(request.request_id)
            if cap is not None and cap < num_new_tokens:
                num_new_tokens = cap
        return super()._mamba_block_aligned_split(
            request,
            num_new_tokens,
            num_new_local_computed_tokens,
            num_external_computed_tokens,
        )

    def _reserve_prefill_lookahead(
        self, request: Request, num_computed_tokens: int, num_new_tokens: int
    ) -> int:
        # Scheduler.schedule() calls this on every request it is about to
        # schedule, in both queues, after its token budget limits, so it is
        # where the step's attention work is tracked and capped.
        if self._in_schedule:
            if self._running_chunk_caps:
                cap = self._running_chunk_caps.get(request.request_id)
                if cap is not None and cap < num_new_tokens:
                    num_new_tokens = cap
            if (
                self._cross_rank_decode_hold_active
                and num_computed_tokens < request.num_tokens - 1
            ):
                return 0
            if (
                num_new_tokens > 0
                and self.promote_warm_waiting
                and self._inflight_prefills
                and (self.scheduler_reserve_full_isl or self.connector is not None)
                and request.status == RequestStatus.WAITING
                and request.num_preemptions == 0
                and request.num_computed_tokens == 0
                and not self._holds_kv_blocks(request)
            ):
                free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
                reserved_async_blocks = (
                    self._inflight_prefill_reserved_blocks()
                    + self._spec_decode_step_blocks
                )
                fits, _, _ = self._waiting_req_fits_kv(
                    request, free_blocks, reserved_async_blocks
                )
                if not fits:
                    return 0
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
        interval = max(
            1, int(getattr(self.scheduler_config, "prefill_schedule_interval", 1) or 1)
        )
        delay = (1 << min(failures - 1, _MAX_BACKOFF_SHIFT)) * interval
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
        """The request upstream's WAITING pass admits for token compute next."""
        free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
        for r in self.kv_holding_waiting:
            if self._is_ready_waiting_request(r) and self._kv_holding_req_fits(
                r, free_blocks
            ):
                return r
        reserved_async_blocks = (
            self._inflight_prefill_reserved_blocks() + self._spec_decode_step_blocks
        )
        for r in self.waiting:
            if not self._is_ready_waiting_request(r):
                continue
            if r.num_computed_tokens == 0 and (
                self.connector is not None
                or (
                    self.promote_warm_waiting
                    and self.scheduler_reserve_full_isl
                    and self._inflight_prefills
                )
            ):
                fits, local_hit, est_hit = self._waiting_req_fits_kv(
                    r, free_blocks, reserved_async_blocks
                )
                if self.connector is not None and est_hit > local_hit:
                    # External async loads (`load_kv_async=True`) consume 0
                    # token_budget and `continue` in Scheduler.schedule() if
                    # they fit KV, or `break` if they do not fit.
                    if not fits:
                        return None
                    continue
                if (
                    not fits
                    and self.promote_warm_waiting
                    and self._inflight_prefills
                    and r.status == RequestStatus.WAITING
                    and r.num_preemptions == 0
                ):
                    continue
            return r
        return None

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
        if (
            self.session_priority
            and target_remaining >= 32768
            and est_hit_tokens < _WARM_FOLLOWUP_MIN_HIT_TOKENS
        ):
            return [], None
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
        if self.session_priority and not self._attn_cap_on and target_remaining < 8192:
            self._share_remaining_budget_with_held(
                kept=kept,
                held=held,
                alloc=alloc,
                left=left,
                lptt=lptt,
                target=target,
                target_need=need,
                hit_blocks=hit_blocks,
                hit_tokens=hit_tokens,
                kv_admit_tokens=kv_admit_tokens,
            )
        return held, target

    def _share_remaining_budget_with_held(
        self,
        *,
        kept: list[Request],
        held: list[Request],
        alloc: dict[str, int],
        left: int,
        lptt: int,
        target: Request,
        target_need: int,
        hit_blocks: KVCacheBlocks,
        hit_tokens: int,
        kv_admit_tokens: int,
    ) -> None:
        """Let held running prefill chunks fill unused step budget past short admissions."""
        align_unit = max(
            int(getattr(self, "hash_block_size", 0) or self.cache_config.block_size or 1),
            1,
        )
        free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
        reserved_async_blocks = (
            self._inflight_prefill_reserved_blocks() + self._spec_decode_step_blocks
        )
        short_reserve = target_need
        seen_target = False
        extra_slots = 0
        max_extra = max(
            0,
            self.max_num_active_reqs
            - (len(kept) + 1 + self.num_waiting_for_streaming_input),
        )
        if max_extra > 0:
            for req_q in chain(self.kv_holding_waiting, self.waiting):
                if not seen_target:
                    if req_q.request_id == target.request_id:
                        seen_target = True
                    continue
                if extra_slots >= max_extra or short_reserve >= left:
                    break
                if not self._is_ready_waiting_request(req_q):
                    continue
                if req_q.num_computed_tokens > 0:
                    if not self._kv_holding_req_fits(req_q, free_blocks):
                        break
                    rem_q = max(req_q.num_tokens - req_q.num_computed_tokens, 1)
                else:
                    fits_q, loc_q, est_q = self._waiting_req_fits_kv(
                        req_q, free_blocks, reserved_async_blocks
                    )
                    if self.connector is not None and est_q > loc_q:
                        if not fits_q:
                            break
                        continue
                    if not fits_q:
                        if (
                            self.promote_warm_waiting
                            and self._inflight_prefills
                            and req_q.status == RequestStatus.WAITING
                            and req_q.num_preemptions == 0
                        ):
                            continue
                        break
                    rem_q = max(req_q.num_tokens - est_q, 1)
                if rem_q >= 8192:
                    break
                need_q = min(rem_q, lptt) if lptt > 0 else rem_q
                if short_reserve + need_q > left:
                    break
                short_reserve += need_q
                extra_slots += 1

        slack = left - short_reserve
        while (
            held
            and slack >= align_unit
            and len(kept) + 1 + self.num_waiting_for_streaming_input
            < self.max_num_active_reqs
        ):
            r_share = held[-1]
            raw_cap = min(self._remaining(r_share), slack)
            if lptt > 0:
                raw_cap = min(raw_cap, lptt)
            aligned_end = (
                (r_share.num_computed_tokens + raw_cap) // align_unit
            ) * align_unit
            share_tokens = aligned_end - r_share.num_computed_tokens
            if self.need_mamba_block_aligned_split and share_tokens > 0:
                share_tokens = super()._mamba_block_aligned_split(r_share, share_tokens)
            share_tokens = super()._reserve_prefill_lookahead(
                r_share, r_share.num_computed_tokens, share_tokens
            )
            if share_tokens < align_unit:
                break
            trial_kept = kept + [r_share]
            trial_alloc = {**alloc, r_share.request_id: share_tokens}
            if not self._kv_fits_hold(
                trial_kept,
                trial_alloc,
                target,
                hit_blocks,
                hit_tokens,
                kv_admit_tokens,
            ):
                break
            held.pop()
            kept.append(r_share)
            alloc[r_share.request_id] = share_tokens
            self._running_chunk_caps[r_share.request_id] = share_tokens
            slack -= share_tokens

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

        target_holds_kv = any(
            r.request_id == target.request_id for r in self.kv_holding_waiting
        )
        if target_holds_kv:
            # `target` already holds its external-hit KV blocks in HBM; match
            # Scheduler.schedule() / allocate_slots(reserved_blocks=0) so ready
            # block-holders can hold back a cold running prefill and drain.
            needed = sum(
                blocks_for(r, r.num_computed_tokens, alloc[r.request_id], empty_blocks)
                for r in kept
                if r.request_id in alloc
            )
            reserved = self._spec_decode_step_blocks
        else:
            needed = sum(
                blocks_for(r, r.num_computed_tokens, alloc[r.request_id], empty_blocks)
                for r in kept
                # In-flight prefills are covered by the reservation below.
                if r.request_id in alloc and r not in self._inflight_prefills
            )
            reserved = (
                self._inflight_prefill_reserved_blocks() + self._spec_decode_step_blocks
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
        free = self.kv_cache_manager.block_pool.get_num_free_blocks()
        return free - reserved >= needed
