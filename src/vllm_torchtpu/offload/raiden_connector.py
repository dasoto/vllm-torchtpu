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
"""Connector wiring for the store-backed Raiden KV offload path.

`TPURaidenOffloadingConnector` subclasses vLLM's `OffloadingConnector` and
orchestrates controller-driven KV offloading via Raiden's C++ `KVCacheStore`.
Enable it via:

    --kv-transfer-config '{
        "kv_connector": "TPURaidenOffloadingConnector",
        "kv_connector_module_path": "vllm_torchtpu.offload.raiden_connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"cpu_bytes_to_use": ...}
    }'

Core mechanisms:

- Save fence: Guarantees background store jobs read HBM only after the forward
  pass writing those KV blocks completes, without inducing device idle bubbles.
  Store jobs ship as `fence_job_ids`. Each rank enqueues KV pool writes
  asynchronously, records a TPU event covering them, and polls for device
  retirement across engine steps before acknowledging via `fenced_jobs`.
  The scheduler launches `save()` only after all ranks have acknowledged.

- Completion echoes: Terminal jobs discovered by the scheduler manager ship
  down to workers, which echo them back as `completed_jobs` / `finished_recving`
  to unblock waiting requests and maintain reference counting. Failed loads
  are also reported by request as `failed_recving`, so the scheduler
  recomputes or fails them, per `kv_load_failure_policy`. Block ids would
  be ambiguous here: each KV cache group numbers its blocks independently,
  and vLLM rejects block-level failures on multi-group layouts.

- Memory reuse (jobs_to_flush): Handled entirely scheduler-side. Fence-pending
  stores are cancelled before launching, while active transfers are drained
  synchronously before shipping metadata, preventing torn reads upon block reuse.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (
    config as offloading_config_module,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
    LookupResult,
    OffloadKey,
    OffloadPolicy,
)
from vllm.v1.outputs import KVConnectorOutput

from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.offload.raiden_store import (
    AdmissionOp,
    RaidenLoadStoreSpec,
    RaidenOffloadingManager,
    TPURaidenStoreOffloadingSpec,
)

if TYPE_CHECKING:
    from vllm.distributed.kv_events import KVCacheEvent

logger = init_logger(__name__)


def _enqueue_pool_writes(pool_tensors: list[torch.Tensor]) -> None:
    """Asynchronously dispatches graph execution for KV pool buffers without blocking.

    Triggers compilation and submission of any deferred operations targeting the
    pool buffers so that subsequent stream events order behind them. Performs a
    non-blocking device synchronization (`wait=False`), which dispatches no new
    work when the writes are already enqueued.
    """
    from torch_tpu._internal import sync

    sync.synchronize(pool_tensors, wait=False)


def _record_fence_event() -> torch.Event:
    """Records a TPU stream event to track completion of enqueued device operations.

    The event orders after work already submitted to the stream, so callers
    enqueue the writes they need fenced (via `_enqueue_pool_writes`) before
    recording.
    """
    event = torch.tpu.Event()
    event.record()
    return event


def _event_is_retired(event: torch.Event) -> bool:
    """Non-blockingly verifies fence event completion and surfaces deferred errors.

    `event.query()` polls device status without blocking the host thread. Once
    execution terminates, calling `event.synchronize()` returns immediately and
    raises any deferred asynchronous runtime errors before acknowledging the fence.
    """
    if not event.query():
        return False
    event.synchronize()
    return True


@dataclass
class RaidenOffloadingConnectorMetadata(OffloadingConnectorMetadata):
    """Scheduler -> worker metadata.

    Transfers are controller-driven (load_jobs and store_jobs are empty);
    extra fields carry fence requests and completion echoes.
    """

    # Store jobs awaiting device fence from all ranks before launching save.
    fence_job_ids: set[int] = field(default_factory=set)
    # Completed store jobs to echo back to the scheduler state machine.
    finished_store_job_ids: list[int] = field(default_factory=list)
    # Completed load jobs (job_id -> req_id) to echo back and unblock requests.
    finished_load_jobs: dict[int, str] = field(default_factory=dict)
    # Requests whose load failed, a subset of finished_load_jobs' values,
    # reported as failed_recving; the scheduler recomputes or fails them per
    # kv_load_failure_policy.
    failed_load_req_ids: set[str] = field(default_factory=set)


@dataclass
class RaidenOffloadingWorkerMetadata(OffloadingWorkerMetadata):
    """Worker -> scheduler metadata: contains completed-job echoes and
    per-rank fence acks."""

    fenced_jobs: dict[int, int] = field(default_factory=dict)

    def aggregate(self, other):
        assert isinstance(other, RaidenOffloadingWorkerMetadata)
        merged_completed = dict(self.completed_jobs)
        for job_id, v in other.completed_jobs.items():
            merged_completed[job_id] = merged_completed.get(job_id, 0) + v
        merged_fenced = dict(self.fenced_jobs)
        for job_id, v in other.fenced_jobs.items():
            merged_fenced[job_id] = merged_fenced.get(job_id, 0) + v
        # completed_jobs and transfer_stats are inherited from vLLM base dataclass.
        return RaidenOffloadingWorkerMetadata(
            # pyrefly: ignore[unexpected-keyword]
            completed_jobs=merged_completed,
            # pyrefly: ignore[unexpected-keyword]
            transfer_stats=self.transfer_stats.aggregate(other.transfer_stats),
            fenced_jobs=merged_fenced,
        )


@dataclass
class _ParkedStore:
    """A store job parked until every worker rank acknowledges the device fence."""

    keys: list[OffloadKey]
    device_block_ids: list[int]
    acks: int = 0


class TPURaidenOffloadingScheduler(OffloadingConnectorScheduler):
    """Scheduler for the store-backed offload path.

    Strips per-rank transfer jobs and orchestrates the store engine directly
    via RaidenOffloadingManager.
    """

    def __init__(
        self,
        spec: TPURaidenStoreOffloadingSpec,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(spec, vllm_config, kv_cache_config)
        if self.config.sync_load_budget_bytes:
            raise ValueError(
                "TPURaidenOffloadingConnector does not support "
                "sync_load_budget_bytes: its loads always complete asynchronously."
            )
        assert isinstance(self.manager, RaidenOffloadingManager)
        self._raiden_manager: RaidenOffloadingManager = self.manager
        self._fence_pending: dict[int, _ParkedStore] = {}

        # Upstream OffloadingConnectorScheduler disables supports_partial_tail
        # whenever decode_context_parallel_size > 1 because CPU canonical-layout
        # workers de-interleave token positions across DCP shards. In contrast,
        # TPURaidenOffloadingConnector transfers opaque whole physical KV blocks
        # rank-for-rank across all DCP workers (get_required_kvcache_layout -> None),
        # so DCP-sharded full-attention blocks plus Mamba CoW boundary blocks at
        # hash-aligned interior positions preserve exact per-rank KV layout.
        non_cow_block_sizes = {
            c.tokens_per_block
            for c in self.config.kv_group_configs
            if not c.requires_cow_source
        }
        non_cow_block_size = (
            next(iter(non_cow_block_sizes)) if len(non_cow_block_sizes) == 1 else 0
        )
        has_partial_recurrent_group = any(
            c.requires_cow_source and c.tokens_per_block >= self.config.tokens_per_hash
            for c in self.config.kv_group_configs
        )
        if (
            not self.config.supports_partial_tail
            and self.config.dcp_world_size > 1
            and self.config.blocks_per_chunk == 1
            and non_cow_block_size > self.config.tokens_per_hash
            and has_partial_recurrent_group
            and all(
                c.sliding_window_size_in_chunks is None
                or (
                    c.requires_cow_source
                    and non_cow_block_size % c.tokens_per_block == 0
                )
                for c in self.config.kv_group_configs
            )
        ):
            from vllm.v1.core.kv_cache_coordinator import eagle_proof_margin

            self.config = self.config._replace(supports_partial_tail=True)
            self._partial_tail_block_size = non_cow_block_size
            alignment = kv_cache_config.cache_hit_alignment_tokens
            partial_hash_hits = (
                alignment is not None and alignment < self._partial_tail_block_size
            )
            self._eagle_proof_margins = {
                c.group_idx: eagle_proof_margin(
                    self._partial_tail_block_size,
                    self.config.tokens_per_hash,
                    partial_hash_hits
                    and c.manager_cls.supports_fine_grained_hash_lookup,
                )
                for c in self.config.kv_group_configs
                if c.is_eagle_group and not c.requires_cow_source
            }
            self._max_eagle_proof_margin = max(
                self._eagle_proof_margins.values(), default=0
            )
            logger.info(
                "TPURaidenOffloadingScheduler: enabled supports_partial_tail "
                "under DCP=%d (partial_tail_block_size=%d, tokens_per_hash=%d).",
                self.config.dcp_world_size,
                self._partial_tail_block_size,
                self.config.tokens_per_hash,
            )

    def _lookup(
        self,
        req_status,
        max_num_new_tokens: int | None = None,
    ) -> int | None:
        hit = super()._lookup(req_status, max_num_new_tokens)
        if (
            hit is None
            or not self.config.supports_partial_tail
            or self._partial_tail_block_size <= 0
            or self.config.retention_interval != 0
            or not self._cow_source_groups
        ):
            return hit

        # Under sparse Mamba retention (`prefix_cache_retention_interval == 0`),
        # Mamba states are only checkpointed and offloaded at turn prompt
        # boundaries (`boundary % tokens_per_hash == 0`), not at interior
        # full-attention super-block boundaries (`block_size = 8,192`).
        # Upstream `_lookup_complete_chunks` intersects full-attention complete
        # chunks with Mamba complete chunks, collapsing `complete_hit` to 0
        # whenever no prior turn ended on an exact `block_size` multiple, and
        # `_lookup` then searches partial tails only within one `block_size`
        # past `local_tokens + complete_hit`. Scan full-attention complete
        # chunks directly to establish the valid prefix horizon and search
        # `tokens_per_hash`-aligned partial-tail boundaries down from there.
        saved_partial = req_status.partial_tail_boundary
        local_tokens = req_status.num_locally_computed_tokens
        block_size = self._partial_tail_block_size
        tokens_per_hash = self.config.tokens_per_hash
        if local_tokens % block_size != 0:
            return hit

        best_boundary = local_tokens + hit
        max_boundary = req_status.req.num_prompt_tokens - 1
        if self._max_eagle_proof_margin:
            max_boundary = min(
                max_boundary,
                len(req_status.req.block_hashes) * tokens_per_hash
                - self._max_eagle_proof_margin,
            )
        if max_num_new_tokens is not None:
            max_boundary = min(max_boundary, local_tokens + max_num_new_tokens)
        if req_status.max_load_tokens is not None:
            max_boundary = min(
                max_boundary, local_tokens + req_status.max_load_tokens
            )
        max_boundary = (max_boundary // tokens_per_hash) * tokens_per_hash
        if max_boundary <= best_boundary:
            return hit

        fa_pairs = [
            (group_config, group_state)
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            )
            if not group_config.requires_cow_source
            and group_config.sliding_window_size_in_chunks is None
        ]
        if not fa_pairs or any(
            gc.tokens_per_chunk != block_size for gc, _ in fa_pairs
        ):
            return hit

        start_chunk = local_tokens // block_size
        max_fa_chunks = min(
            max_boundary // block_size,
            min(len(gs.offload_keys) for _, gs in fa_pairs),
        )
        chunks_being_loaded = self._chunks_being_loaded or ()
        fa_confirmed_end = start_chunk
        fa_chunk_pending: dict[int, bool] = {}
        any_pending = False
        for chunk_idx in range(start_chunk, max_fa_chunks):
            chunk_miss = False
            chunk_pend = False
            for _, gs in fa_pairs:
                key = gs.offload_keys[chunk_idx]
                res = self.manager.lookup(key, req_status.req_context)
                if res is LookupResult.MISS:
                    chunk_miss = True
                    break
                chunk_pend |= (res is not LookupResult.HIT) or (
                    key in chunks_being_loaded
                )
            if chunk_miss:
                break
            fa_chunk_pending[chunk_idx] = chunk_pend
            any_pending |= chunk_pend
            fa_confirmed_end = chunk_idx + 1

        search_hi = min(
            max_boundary,
            (fa_confirmed_end + 1) * block_size - tokens_per_hash,
        )
        if search_hi <= best_boundary:
            if any_pending and hit == 0:
                req_status.partial_tail_boundary = None
                return None
            return hit

        for boundary in range(search_hi, best_boundary, -tokens_per_hash):
            if boundary % block_size == 0:
                continue
            needed_fa_end = boundary // block_size
            if needed_fa_end > fa_confirmed_end:
                continue
            bridge_pending = any(
                fa_chunk_pending.get(c, False)
                for c in range(start_chunk, needed_fa_end)
            )
            if chunks_being_loaded and not bridge_pending:
                for gc in self.config.kv_group_configs:
                    kt = self._partial_tail_key_tokens(gc, boundary)
                    pkey = self._make_boundary_key(
                        req_status.req, gc.group_idx, kt, req_status.req_context
                    )
                    if pkey in chunks_being_loaded:
                        bridge_pending = True
                        break
            pt_hit, pt_pending = self._lookup_partial_tail(
                req_status, boundary, bridge_pending
            )
            if pt_hit:
                for gc, gs in fa_pairs:
                    rec_from = max(start_chunk, gs.lookup_hit_end_chunk_idx)
                    for c in range(rec_from, needed_fa_end):
                        self._events_tracker.record_lookup(
                            req_status.req, gc, c, gs.offload_keys[c]
                        )
                req_status.partial_tail_boundary = boundary
                return boundary - local_tokens
            any_pending |= pt_pending

        req_status.partial_tail_boundary = saved_partial
        if any_pending and hit == 0:
            req_status.partial_tail_boundary = None
            return None
        return hit

    def update_state_after_alloc(
        self, request, blocks, num_external_tokens: int
    ) -> None:
        req_status = self._req_status.get(request.request_id)
        if num_external_tokens == 0 and req_status is not None:
            req_status.partial_tail_boundary = None
        had_partial_tail = (
            num_external_tokens > 0
            and req_status is not None
            and req_status.partial_tail_boundary is not None
        )
        super().update_state_after_alloc(request, blocks, num_external_tokens)
        if (
            had_partial_tail
            and req_status is not None
            and req_status.offloading_context.policy == OffloadPolicy.CHUNK_LEVEL
        ):
            # Upstream sets next_stored_chunk_idx = cdiv(num_cached_tokens,
            # tokens_per_chunk), skipping the partially filled chunk once
            # resumed prefill completes it and leaving a permanent hole in host
            # prefix lookup for subsequent turns. Only complete chunks
            # (num_cached_tokens // tokens_per_chunk) are already stored.
            num_cached_tokens = (
                req_status.num_locally_computed_tokens + num_external_tokens
            )
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                group_state.next_stored_chunk_idx = (
                    num_cached_tokens // group_config.tokens_per_chunk
                )

    def _build_partial_tail_store_jobs(self, scheduler_output: SchedulerOutput):
        block_state = scheduler_output.kv_connector_block_state
        handoffs = block_state.boundary_state_offloads if block_state else None
        if (
            handoffs
            and self.config.supports_partial_tail
            and self._partial_tail_block_size > 0
        ):
            filtered: dict[str, list[tuple[int, int, int]]] = {}
            changed = False
            block_size = self._partial_tail_block_size
            tokens_per_hash = self.config.tokens_per_hash
            for req_id, entries in handoffs.items():
                req_status = self._req_status.get(req_id)
                if req_status is None:
                    changed = True
                    continue
                req = req_status.req
                prompt_cap = min(
                    req.num_prompt_tokens,
                    req_status.max_offload_tokens or req.num_prompt_tokens,
                )
                # Under DCP > 1, FullAttention chunks span `block_size` tokens
                # (e.g. 13,312 = 8 * 1,664) while Mamba align blocks emit a
                # hand-off every `tokens_per_hash` (1,664) tokens. Keep:
                # 1. Super-block-aligned hand-offs (`boundary % block_size == 0`)
                #    so `_build_aligned_boundary_store_jobs` stores the Mamba
                #    state matching each complete FullAttention offload chunk.
                # 2. At most ONE sub-super-block partial-tail boundary (matching
                #    upstream's `len(boundaries) == 1` invariant), restricted to
                #    the prompt's reusable tail window (or an explicit
                #    `shared_prefix_boundary` junction) so intermediate chunked
                #    prefill steps neither amplify host DRAM / PCIe stores by
                #    `dcp_world_size`x nor crash when an async-load completion
                #    and a resumed prefill chunk emit two sub-super-block
                #    boundaries in the same step.
                prompt_tail = (prompt_cap // tokens_per_hash) * tokens_per_hash
                prompt_tail_floor = (
                    prompt_tail - tokens_per_hash
                    if prompt_tail > 0 and prompt_cap % tokens_per_hash == 0
                    else prompt_tail
                )
                aligned = [e for e in entries if e[2] % block_size == 0]
                partial_candidates = [
                    e
                    for e in entries
                    if e[2] % block_size != 0
                    and 0 < e[2] <= prompt_cap
                    and (
                        e[2] >= prompt_tail_floor
                        or (
                            req.shared_prefix_boundary > 0
                            and e[2] == req.shared_prefix_boundary
                        )
                    )
                ]
                chosen_partial: list[tuple[int, int, int]] = []
                if partial_candidates:
                    by_boundary: dict[int, dict[int, tuple[int, int, int]]] = {}
                    for e in partial_candidates:
                        by_boundary.setdefault(e[2], {})[e[0]] = e
                    valid_boundaries = [
                        b
                        for b, group_map in by_boundary.items()
                        if self._cow_source_groups.issubset(group_map)
                    ]
                    if valid_boundaries:
                        best_boundary = max(valid_boundaries)
                        chosen_partial = list(by_boundary[best_boundary].values())
                kept = aligned + chosen_partial
                if len(kept) != len(entries):
                    changed = True
                if kept:
                    filtered[req_id] = kept
            if changed:
                from dataclasses import replace

                scheduler_output = replace(
                    scheduler_output,
                    kv_connector_block_state=replace(
                        block_state, boundary_state_offloads=filtered
                    ),
                )
        return super()._build_partial_tail_store_jobs(scheduler_output)

    def get_num_new_matched_tokens(
        self,
        request,
        num_computed_tokens: int,
        max_num_new_tokens: int | None = None,
    ) -> tuple[int | None, bool]:
        res = super().get_num_new_matched_tokens(
            request, num_computed_tokens, max_num_new_tokens
        )
        # Upstream OffloadingConnectorScheduler skips local HBM-hit chunks in
        # _lookup and never calls manager.touch(), leaving early prefix chunks
        # of active multi-turn sessions at the tail of Raiden's host LRU.
        # Refresh resident host-pool entries covering the local hit once
        # admission settles (res[0] is not None), inside the step lock opened
        # by super().get_num_new_matched_tokens().
        if (
            res[0] is not None
            and num_computed_tokens > 0
            and not request.skip_reading_prefix_cache
        ):
            req_status = self._req_status.get(request.request_id)
            if req_status is not None:
                local_keys: list[OffloadKey] = []
                has_local_partial_tail = (
                    self.config.supports_partial_tail
                    and self._partial_tail_block_size > 0
                    and num_computed_tokens % self._partial_tail_block_size != 0
                    and num_computed_tokens % self.config.tokens_per_hash == 0
                )
                for group_config, group_state in zip(
                    self.config.kv_group_configs, req_status.group_states
                ):
                    n_local_chunks = (
                        num_computed_tokens // group_config.tokens_per_chunk
                    )
                    if n_local_chunks > 0:
                        local_keys.extend(group_state.offload_keys[:n_local_chunks])
                    if has_local_partial_tail:
                        key_tokens = self._partial_tail_key_tokens(
                            group_config, num_computed_tokens
                        )
                        if key_tokens <= len(request.block_hashes) * self.config.tokens_per_hash:
                            local_keys.append(
                                self._make_boundary_key(
                                    request,
                                    group_config.group_idx,
                                    key_tokens,
                                    req_status.req_context,
                                )
                            )
                if local_keys:
                    self._raiden_manager.touch(local_keys, req_status.req_context)
        return res

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        meta = super().build_connector_meta(scheduler_output)
        assert isinstance(meta, OffloadingConnectorMetadata)

        manager = self._raiden_manager

        # Park new store jobs: wait for all ranks to complete device fence.
        new_fence_ids: set[int] = set()
        for job_id, job in meta.store_jobs.items():
            src_spec, dst_spec = job.src_spec, job.dst_spec
            assert isinstance(src_spec, GPULoadStoreSpec)
            assert isinstance(dst_spec, RaidenLoadStoreSpec)
            device_block_ids = src_spec.block_ids.tolist()
            assert len(device_block_ids) == len(dst_spec.keys), (
                len(device_block_ids),
                len(dst_spec.keys),
            )
            self._fence_pending[job_id] = _ParkedStore(
                keys=dst_spec.keys, device_block_ids=device_block_ids
            )
            new_fence_ids.add(job_id)

        # Submit load jobs immediately: destination HBM blocks are freshly
        # allocated and safe to write into.
        for job_id, job in meta.load_jobs.items():
            src_spec, dst_spec = job.src_spec, job.dst_spec
            assert isinstance(src_spec, RaidenLoadStoreSpec)
            assert isinstance(dst_spec, GPULoadStoreSpec)
            device_block_ids = dst_spec.block_ids.tolist()
            assert len(device_block_ids) == len(src_spec.keys), (
                len(device_block_ids),
                len(src_spec.keys),
            )
            manager.submit_load_job(
                job_id,
                src_spec.keys,
                device_block_ids,
                job.req_id,
                pinned=src_spec.pinned,
            )

        # Handle jobs_to_flush (source blocks are about to be recycled):
        # - Cancel fence-pending stores before they start.
        # - Synchronously drain launched stores so DMA finishes before blocks
        #   are overwritten.
        finished_store_ids: list[int] = []
        finished_loads: dict[int, str] = {}
        failed_load_req_ids: set[str] = set()
        flush_ids = meta.jobs_to_flush or set()
        launched_flush_ids: set[int] = set()
        for job_id in flush_ids:
            parked = self._fence_pending.pop(job_id, None)
            if parked is not None:
                manager.on_store_job_cancelled(parked.keys)
                finished_store_ids.append(job_id)
                # Avoid useless TPU sync if created and cancelled in the same step.
                new_fence_ids.discard(job_id)
                logger.debug(
                    "TPURaidenOffloadingConnector: cancelled fence-pending "
                    "store job %d (source blocks reused)",
                    job_id,
                )
            elif manager.is_job_launched(job_id):
                launched_flush_ids.add(job_id)
        drained = manager.drain_jobs(launched_flush_ids) if launched_flush_ids else []

        # Collect completed/drained jobs for the worker echo channel.
        for finished_job in list(drained) + manager.poll_finished_jobs():
            if finished_job.op == AdmissionOp.STORE:
                finished_store_ids.append(finished_job.job_id)
            else:
                assert finished_job.req_id is not None
                finished_loads[finished_job.job_id] = finished_job.req_id
                if not finished_job.success:
                    failed_load_req_ids.add(finished_job.req_id)

        # Return empty load_jobs/store_jobs: workers act as passive DMA endpoints.
        return RaidenOffloadingConnectorMetadata(
            load_jobs={},  # pyrefly: ignore[unexpected-keyword]
            store_jobs={},  # pyrefly: ignore[unexpected-keyword]
            # pyrefly: ignore[unexpected-keyword]
            jobs_to_flush=meta.jobs_to_flush,
            fence_job_ids=new_fence_ids,
            finished_store_job_ids=finished_store_ids,
            finished_load_jobs=finished_loads,
            failed_load_req_ids=failed_load_req_ids,
        )

    def update_connector_output(self, connector_output: KVConnectorOutput):
        meta = connector_output.kv_connector_worker_meta
        if isinstance(meta, RaidenOffloadingWorkerMetadata) and meta.fenced_jobs:
            for job_id, count in meta.fenced_jobs.items():
                parked = self._fence_pending.get(job_id)
                if parked is None:
                    # Cancelled while the acks were in flight.
                    continue
                parked.acks += count
                assert parked.acks <= self.config.num_workers
                if parked.acks == self.config.num_workers:
                    del self._fence_pending[job_id]
                    self._raiden_manager.submit_store_job(
                        job_id, parked.keys, parked.device_block_ids
                    )
        super().update_connector_output(connector_output)


class TPURaidenOffloadingConnector(OffloadingConnector):
    """Top-level facade connector for the store-backed (V2) raiden KV offload path."""

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str | None:
        """No layout preference, unlike the `OffloadingConnector` this extends.

        Upstream asks for "LBHNC" ("BLHNC" with HiSparse), a layout the TPU
        KV cache does not use; these pages are opaque bytes to this connector.
        """
        return None

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        # Opt in to uniform cross-layer KV cache allocation when block-major layout is
        # enabled. Satisfies `KVConnectorModelRunnerMixin.use_uniform_kv_cache()`
        # preconditions to enable bundled tensor allocation in the TPU model runner.
        return bool(tpu_envs.VLLM_TPU_BLOCK_MAJOR_KV)

    @property
    def supports_divergent_local_hybrid_hits(self) -> bool:
        """Complete a deeper full-attention local hit, behind the Stage-3 flag.

        Upstream offload lookup treats a Mamba group as a one-chunk window, so
        a hit from this store ends where it holds the recurrent state for its
        last chunk: a load restores the state the local hit lacks, and a miss
        lets the scheduler reconcile to the common boundary. Decode composes
        this connector with TPURaidenConnector in TPUMultiConnector, which
        needs every child to agree before Stage-3 suffix loads can run. Gated
        on the prefix-aware flag so standalone offload keeps today's lookup.
        """
        # Deferred: tpu_connector is heavy and only this property needs it.
        from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
            _prefix_aware_load_enabled,
        )

        return _prefix_aware_load_enabled()

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        # Skip OffloadingConnector.__init__: default factory cannot carry
        # vllm_config / kv_cache_config directly into the spec.
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)

        # The raiden singleton worker cannot coexist with this connector: the
        # disagg stack runs its own control plane in the same process and the
        # two collide on ports.
        if not tpu_envs.RAIDEN_DISABLE_SINGLETON_WORKER:
            raise ValueError(
                "TPURaidenOffloadingConnector requires the raiden singleton "
                "worker to be disabled (it collides with the in-process "
                "control plane's ports), but RAIDEN_DISABLE_SINGLETON_WORKER "
                "was explicitly set to off. Unset it or set it to '1'."
            )
        os.environ["RAIDEN_DISABLE_SINGLETON_WORKER"] = "1"

        # Call through module to preserve PCP-aware patch on build_offloading_config.
        offloading_config = offloading_config_module.build_offloading_config(
            vllm_config, kv_cache_config
        )
        spec = TPURaidenStoreOffloadingSpec(
            offloading_config, vllm_config, kv_cache_config
        )

        # pyrefly: ignore[bad-override-mutable-attribute]
        self.connector_scheduler: TPURaidenOffloadingScheduler | None = None
        self.connector_worker: OffloadingConnectorWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = TPURaidenOffloadingScheduler(
                spec, vllm_config, kv_cache_config
            )
        elif role == KVConnectorRole.WORKER:
            # Worker wrapper is only used for KV buffer registration; job dicts
            # remain empty.
            self.connector_worker = OffloadingConnectorWorker(
                spec, vllm_config, kv_cache_config
            )
            # Track per-step fence acknowledgments and completion echoes for
            # worker metadata.
            self._fenced_jobs: dict[int, int] = {}
            self._completed_jobs: dict[int, int] = {}
            # Registered KV pool byte views whose deferred writes are dispatched before
            # recording fence events.
            self._pool_sync_tensors: list[torch.Tensor] = []
            # In-flight fence events awaiting device retirement before acknowledging
            # store jobs, recorded as pairs of (fence_job_ids, event).
            self._pending_fences: list[tuple[set[int], torch.Event]] = []

    # -- scheduler-side hooks ----------------------------------------------
    def reset_cache(self) -> bool | None:
        # External cache reset is unsupported by the Raiden store. Report
        # failure through the endpoint's documented channel instead of raising an
        # exception here
        logger.warning(
            "TPURaidenOffloadingConnector: external cache reset is not "
            "supported by the Raiden offload store; reporting failure."
        )
        return False

    def take_events(self) -> Iterable[KVCacheEvent]:
        if self.connector_scheduler is not None:
            return self.connector_scheduler.take_events()
        return []

    # -- worker-side hooks -------------------------------------------------
    def register_kv_caches(
        self, kv_caches: Mapping[str, torch.Tensor | list[torch.Tensor]]
    ):
        """Register the TPU UBP as zero-copy canonical offload tensors.

        vLLM's generic per-layer registration does not support the TPU
        unified block pool layout or Mamba tensor lists, so this
        implementation deduplicates the UBP backing storages and exposes
        each as an `int8` byte view.
        """
        assert self.connector_worker is not None
        num_blocks = self.connector_worker.kv_cache_config.num_blocks

        pool_storages: dict[int, tuple[torch.UntypedStorage, torch.device]] = {}
        for layer_kv_cache in kv_caches.values():
            state_tensors = (
                [layer_kv_cache]
                if isinstance(layer_kv_cache, torch.Tensor)
                else layer_kv_cache
            )
            for tensor in state_tensors:
                storage = tensor.untyped_storage()
                pool_storages.setdefault(storage.data_ptr(), (storage, tensor.device))

        pool_tensors: list[CanonicalKVCacheTensor] = []
        for storage, device in pool_storages.values():
            storage_bytes = storage.nbytes()
            assert storage_bytes % num_blocks == 0, (
                "KV pool storage must hold whole scheduler blocks: "
                f"storage_bytes={storage_bytes}, num_blocks={num_blocks}"
            )
            page_size_bytes = storage_bytes // num_blocks
            int8_view = (
                torch.empty(0, dtype=torch.int8, device=device)
                .set_(storage)
                .view(num_blocks, page_size_bytes)
            )
            pool_tensors.append(
                CanonicalKVCacheTensor(
                    tensor=int8_view, page_size_bytes=page_size_bytes
                )
            )

        # Retain pool tensor references to dispatch their writes during save fences.
        self._pool_sync_tensors = [pool_tensor.tensor for pool_tensor in pool_tensors]

        # One ref list per offloaded group, as vLLM's own registration builds:
        # groups that are not prefix-cacheable are not offloaded.
        group_data_refs = [
            [
                CanonicalKVCacheRef(
                    tensor_idx=idx, page_size_bytes=pool_tensor.page_size_bytes
                )
                for idx, pool_tensor in enumerate(pool_tensors)
            ]
            for _ in offloading_config_module.get_offloading_group_ids(
                self.connector_worker.kv_cache_config
            )
        ]

        self.connector_worker._init_worker(
            CanonicalKVCaches(tensors=pool_tensors, group_data_refs=group_data_refs)
        )

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        # Load failures only travel in get_transfer_results().failed_recving.
        # Returning a failed request here would pass its load as a good one,
        # so refuse loudly; callers must use get_transfer_results().
        results = self.get_transfer_results(finished_req_ids)
        if results.failed_recving:
            raise RuntimeError(
                "TPURaidenOffloadingConnector.get_finished() cannot report the "
                f"failed loads of requests {sorted(results.failed_recving)}; "
                "call get_transfer_results() instead"
            )
        return results.finished_sending, results.finished_recving

    def get_transfer_results(
        self, finished_req_ids: set[str]
    ) -> KVConnectorTransferResults:
        assert self.connector_worker is not None
        meta = self._get_connector_metadata()
        assert isinstance(meta, RaidenOffloadingConnectorMetadata)

        if meta.fence_job_ids:
            # Establish an asynchronous device fence before acknowledging store jobs:
            # 1. Dispatch any deferred pool writes so they precede the fence event.
            # 2. Record a stream event to track completion without host-side blocking.
            assert self._pool_sync_tensors, (
                "KV pool tensors must be registered before the save fence"
            )
            _enqueue_pool_writes(self._pool_sync_tensors)
            self._pending_fences.append(
                (set(meta.fence_job_ids), _record_fence_event())
            )

        if self._pending_fences:
            # Non-blockingly drain retired events and acknowledge completed fences.
            still_pending: list[tuple[set[int], torch.Event]] = []
            for job_ids, event in self._pending_fences:
                if _event_is_retired(event):
                    for job_id in job_ids:
                        self._fenced_jobs[job_id] = 1
                else:
                    still_pending.append((job_ids, event))
            self._pending_fences = still_pending

        finished_recving: set[str] = set()
        for job_id in meta.finished_store_job_ids:
            self._completed_jobs[job_id] = 1
        for job_id, req_id in meta.finished_load_jobs.items():
            self._completed_jobs[job_id] = 1
            finished_recving.add(req_id)

        # A failed request also stays in finished_recving: that is what
        # releases it from WAITING_FOR_REMOTE_KVS.
        return KVConnectorTransferResults(
            finished_recving=finished_recving,
            failed_recving=set(meta.failed_load_req_ids),
        )

    def build_connector_worker_meta(self) -> OffloadingWorkerMetadata | None:
        if not (self._fenced_jobs or self._completed_jobs):
            return None
        # completed_jobs is inherited from vLLM base dataclass.
        meta = RaidenOffloadingWorkerMetadata(
            # pyrefly: ignore[unexpected-keyword]
            completed_jobs=self._completed_jobs,
            fenced_jobs=self._fenced_jobs,
        )
        self._completed_jobs = {}
        self._fenced_jobs = {}
        return meta
