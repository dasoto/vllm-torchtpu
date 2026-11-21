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
"""Cache-aware prefix + load-balanced Data Parallel routing for DPLBAsyncMPClient.

Upstream ``DPLBAsyncMPClient.get_core_engine_for_request`` selects DP engines
solely by queue load (``waiting + running``), with no prefix-cache affinity.
Under multi-turn agentic workloads at ``DP > 1``, consecutive turns of the same
conversation land on different DP engines, causing repeated cold long-context
prefills and ``DP``-fold duplication of session KV caches across engines.

When ``TPU_DP_PREFIX_ROUTING=1`` (default) and ``enable_prefix_caching=True``,
this module tracks rolling token-chunk hashes across DP engines in an LRU
directory and scores each DP engine by combined queue load, KV-cache footprint
balance, and uncached prompt tokens ``(L - cached_tokens[engine])``:
- Shared system prompts naturally warm across all DP engines (the storage
  balance term prevents the first engine that caches a shared system prompt
  from attracting all subsequent cold sessions).
- Multi-turn follow-ups match their full conversation history on the home DP
  engine and stick to it unless that engine is heavily overloaded.
"""

from __future__ import annotations

import array
import hashlib
import sys
import time
from collections import OrderedDict
from functools import wraps

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_CHUNK_TOKENS = 512
_MAX_TRACKED_CHUNKS = 262_144
_RUNNING_TOKEN_WEIGHT = 512.0
_WAITING_TOKEN_WEIGHT = 4096.0
_STORAGE_BALANCE_WEIGHT = 6144.0
_STICKY_LONG_PREFIX_TOKENS = 8_192
_WARM_FOLLOWUP_MAX_UNCACHED = 65_536
_PREFILL_DRAIN_TOKENS_PER_SEC = 900.0


def _rolling_chunk_digests(
    token_ids: list[int], chunk_tokens: int = _DEFAULT_CHUNK_TOKENS
) -> list[bytes]:
    """Compute chained 8-byte BLAKE2b digests for each whole `chunk_tokens` block."""
    num_chunks = len(token_ids) // chunk_tokens
    if num_chunks <= 0:
        return []
    arr = array.array("I", token_ids[: num_chunks * chunk_tokens])
    mv = memoryview(arr).cast("B")
    stride = chunk_tokens * 4
    digests: list[bytes] = []
    prev = b"\x00" * 8
    for idx in range(num_chunks):
        h = hashlib.blake2b(digest_size=8, person=prev)
        h.update(mv[idx * stride : (idx + 1) * stride])
        prev = h.digest()
        digests.append(prev)
    return digests


class _DPPrefixDirectory:
    """Bounded LRU directory mapping chunk digests to owning DP engine indices."""

    def __init__(
        self,
        num_engines: int,
        chunk_tokens: int = _DEFAULT_CHUNK_TOKENS,
        max_entries: int = _MAX_TRACKED_CHUNKS,
    ) -> None:
        self.num_engines = num_engines
        self.chunk_tokens = chunk_tokens
        self.max_entries = max_entries
        self._owners: OrderedDict[bytes, int] = OrderedDict()
        self._engine_counts: list[int] = [0] * num_engines
        self._engine_prefill_backlog: list[float] = [0.0] * num_engines
        self._last_decay_t: float | None = None

    def _decay_prefill_backlog(self, now: float | None = None) -> None:
        cur = time.monotonic() if now is None else now
        if self._last_decay_t is None:
            self._last_decay_t = cur
            return
        dt = max(0.0, cur - self._last_decay_t)
        self._last_decay_t = cur
        if dt <= 0.0:
            return
        drained = dt * _PREFILL_DRAIN_TOKENS_PER_SEC
        for idx in range(self.num_engines):
            prev = self._engine_prefill_backlog[idx]
            if prev > 0.0:
                self._engine_prefill_backlog[idx] = max(0.0, prev - drained)

    def cached_tokens_by_engine(self, digests: list[bytes]) -> list[int]:
        """Return deepest matched prefix length (in tokens) for each DP engine."""
        cached = [0] * self.num_engines
        if not digests or not self._owners:
            return cached
        remaining = self.num_engines
        owners = self._owners
        unit = self.chunk_tokens
        for pos in range(len(digests) - 1, -1, -1):
            mask = owners.get(digests[pos], 0)
            if not mask:
                continue
            tokens = (pos + 1) * unit
            m = mask
            while m:
                lsb = m & -m
                eng_idx = lsb.bit_length() - 1
                if eng_idx < self.num_engines and cached[eng_idx] == 0:
                    cached[eng_idx] = tokens
                    remaining -= 1
                    if remaining == 0:
                        return cached
                m ^= lsb
        return cached

    def record(
        self,
        digests: list[bytes],
        eng_idx: int,
        prompt_len: int = 0,
        now: float | None = None,
    ) -> None:
        if not (0 <= eng_idx < self.num_engines):
            return
        if prompt_len > 0:
            self._decay_prefill_backlog(now)
            matched = self.cached_tokens_by_engine(digests)[eng_idx]
            uncached = max(0, prompt_len - matched)
            if uncached > 0:
                self._engine_prefill_backlog[eng_idx] += float(uncached)
        if not digests:
            return
        bit = 1 << eng_idx
        owners = self._owners
        counts = self._engine_counts
        for d in digests:
            prev = owners.get(d, 0)
            if prev:
                owners.move_to_end(d)
                if not (prev & bit):
                    owners[d] = prev | bit
                    counts[eng_idx] += 1
            else:
                if len(owners) >= self.max_entries:
                    _, evicted_mask = owners.popitem(last=False)
                    m = evicted_mask
                    while m:
                        lsb = m & -m
                        e = lsb.bit_length() - 1
                        if e < self.num_engines and counts[e] > 0:
                            counts[e] -= 1
                        m ^= lsb
                owners[d] = bit
                counts[eng_idx] += 1

    def select_engine(
        self,
        *,
        prompt_token_ids: list[int],
        lb_engines: list[tuple[int, int, float]],
        inflight_by_idx: list[int],
        client_count: int,
        eng_start_index: int,
        now: float | None = None,
    ) -> tuple[int, list[bytes]]:
        self._decay_prefill_backlog(now)
        digests = _rolling_chunk_digests(prompt_token_ids, self.chunk_tokens)
        cached = self.cached_tokens_by_engine(digests)
        prompt_len = len(prompt_token_ids)
        max_cached = max(cached) if cached else 0
        long_warm_session = (
            max_cached >= _STICKY_LONG_PREFIX_TOKENS
            and (prompt_len - max_cached) <= _WARM_FOLLOWUP_MAX_UNCACHED
        )
        uncached_scale = float(max(self.num_engines, 1)) if long_warm_session else 1.0
        avg_blocks = max(sum(self._engine_counts) / max(self.num_engines, 1), 1.0)

        best_idx = eng_start_index % self.num_engines
        best_cost = float(sys.maxsize)
        best_tie = self.num_engines

        for offset in range(self.num_engines):
            idx = (eng_start_index + offset) % self.num_engines
            waiting, running, kv_usage = lb_engines[idx]
            local_inflight = client_count * inflight_by_idx[idx]
            unreported = max(0, local_inflight - (waiting + running))
            eff_waiting = float(waiting + unreported)
            is_warm_home = (
                long_warm_session
                and cached[idx] >= _STICKY_LONG_PREFIX_TOKENS
                and (prompt_len - cached[idx]) <= _WARM_FOLLOWUP_MAX_UNCACHED
            )
            if is_warm_home:
                # Warm long-context follow-ups are promoted ahead of cold waiting
                # prefills by SrptPrefillScheduler; do not spill them to a cold
                # DP engine where recomputing 50K-180K tokens stalls all EP ranks.
                eff_waiting = min(eff_waiting, 1.0)
            eff_running = float(running)
            load_cost = (
                eff_running * _RUNNING_TOKEN_WEIGHT
                + eff_waiting * _WAITING_TOKEN_WEIGHT * (1.0 + 2.0 * kv_usage)
            )
            if not is_warm_home:
                load_cost += self._engine_prefill_backlog[idx]
            uncached = float(max(0, prompt_len - cached[idx])) * uncached_scale
            storage_ratio = self._engine_counts[idx] / avg_blocks
            cost = (
                load_cost
                + uncached
                + storage_ratio * _STORAGE_BALANCE_WEIGHT
                + kv_usage * _STORAGE_BALANCE_WEIGHT
            )
            if cost < best_cost or (cost == best_cost and offset < best_tie):
                best_cost = cost
                best_idx = idx
                best_tie = offset

        return best_idx, digests


def patch_dp_prefix_routing() -> None:
    """Install cache-aware DP request routing on ``DPLBAsyncMPClient``."""
    from vllm.v1.engine import core_client

    cls = getattr(core_client, "DPLBAsyncMPClient", None)
    if cls is None:
        return
    marker = "_tpu_dp_prefix_routing_patch"
    if getattr(cls.get_core_engine_for_request, marker, False):
        return

    original = cls.get_core_engine_for_request

    @wraps(original)
    def get_core_engine_for_request(self, request):
        if (
            not envs.TPU_DP_PREFIX_ROUTING
            or not self.vllm_config.cache_config.enable_prefix_caching
            or len(self.core_engines) <= 1
            or request.data_parallel_rank is not None
            or request.resumable
            or request.request_id in self.reqs_in_flight
            or not request.prompt_token_ids
        ):
            return original(self, request)

        directory: _DPPrefixDirectory | None = getattr(self, "_tpu_dp_prefix_dir", None)
        if directory is None or directory.num_engines != len(self.core_engines):
            block_size = int(self.vllm_config.cache_config.block_size or 512)
            chunk_tokens = max(min(block_size, _DEFAULT_CHUNK_TOKENS), 64)
            directory = _DPPrefixDirectory(
                num_engines=len(self.core_engines),
                chunk_tokens=chunk_tokens,
            )
            self._tpu_dp_prefix_dir = directory
            self._tpu_eng_to_idx = {eng: i for i, eng in enumerate(self.core_engines)}
            logger.info(
                "TPU DP cache-aware prefix routing active across %d DP engines "
                "(chunk_tokens=%d).",
                len(self.core_engines),
                chunk_tokens,
            )

        inflight_by_idx = [self.engine_inflight[eng] for eng in self.core_engines]
        chosen_idx, digests = directory.select_engine(
            prompt_token_ids=request.prompt_token_ids,
            lb_engines=self.lb_engines,
            inflight_by_idx=inflight_by_idx,
            client_count=self.client_count,
            eng_start_index=self.eng_start_index,
        )
        request.data_parallel_rank = chosen_idx
        try:
            engine = original(self, request)
        finally:
            request.data_parallel_rank = None
        actual_idx = self._tpu_eng_to_idx.get(engine, chosen_idx)
        directory.record(
            digests, actual_idx, prompt_len=len(request.prompt_token_ids)
        )
        return engine

    setattr(get_core_engine_for_request, marker, True)
    cls.get_core_engine_for_request = get_core_engine_for_request
    logger.info("Applied TPU patch: cache-aware DP prefix routing.")
