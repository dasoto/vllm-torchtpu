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
from contextlib import contextmanager
import fcntl
import hashlib
import mmap
import os
import struct
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
_PREFILL_DRAIN_TOKENS_PER_SEC = 750.0

_SHM_MAGIC = b"TPUDPPR2"
_SHM_MAX_ENGINES = 64
_SHM_MAX_CLIENTS = 64
_SHM_NUM_SETS = 262_144  # 262,144 sets x 4 ways = 1,048,576 chunk entries (24 MiB)
_SHM_SET_MASK = _SHM_NUM_SETS - 1
_SHM_WAYS = 4
_SHM_ENTRY_STRUCT = struct.Struct("<8sQQ")  # (key_8b, owners_mask_u64, lru_seq_u64)
_SHM_ENTRY_BYTES = _SHM_ENTRY_STRUCT.size  # 24 bytes
_SHM_SET_BYTES = _SHM_WAYS * _SHM_ENTRY_BYTES  # 96 bytes
_SHM_INFLIGHT_OFFSET = 1088  # 64 clients x 64 engines x 4B int32 = 16,384 bytes
_SHM_HEADER_BYTES = 20_480
_SHM_TOTAL_BYTES = _SHM_HEADER_BYTES + _SHM_NUM_SETS * _SHM_SET_BYTES
_ZERO_KEY = b"\x00" * 8
_NONZERO_SENTINEL = b"\x01" + b"\x00" * 7


def _shm_key(digest: bytes) -> bytes:
    return _NONZERO_SENTINEL if digest == _ZERO_KEY else digest


class _SharedMmapPrefixStore:
    """Cross-process `/dev/shm` 4-way set-associative prefix directory for multi-APIServer."""

    def __init__(self, path: str, num_engines: int, chunk_tokens: int) -> None:
        if num_engines > _SHM_MAX_ENGINES:
            raise ValueError(
                f"num_engines={num_engines} exceeds _SHM_MAX_ENGINES={_SHM_MAX_ENGINES}"
            )
        self.path = path
        self.num_engines = num_engines
        self.chunk_tokens = chunk_tokens
        self._row_fmt = f"<{num_engines}i"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        with self._lock():
            st = os.fstat(self._fd)
            if st.st_size != _SHM_TOTAL_BYTES:
                os.ftruncate(self._fd, _SHM_TOTAL_BYTES)
            self._mm = mmap.mmap(self._fd, _SHM_TOTAL_BYTES, access=mmap.ACCESS_WRITE)
            magic, hdr_eng, hdr_chunk = struct.unpack_from("<8sII", self._mm, 0)
            if (
                magic != _SHM_MAGIC
                or hdr_eng != num_engines
                or hdr_chunk != chunk_tokens
            ):
                self._reset_unlocked()

    @contextmanager
    def _lock(self):
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(self._fd, fcntl.LOCK_UN)

    def _reset_unlocked(self) -> None:
        # Preserve live per-client inflight rows when resetting prefix cache.
        saved_inflight = bytes(
            self._mm[
                _SHM_INFLIGHT_OFFSET : _SHM_INFLIGHT_OFFSET
                + _SHM_MAX_CLIENTS * _SHM_MAX_ENGINES * 4
            ]
        )
        had_magic = self._mm[:8] == _SHM_MAGIC
        self._mm.seek(0)
        self._mm.write(b"\x00" * _SHM_TOTAL_BYTES)
        struct.pack_into(
            "<8sIIQdI",
            self._mm,
            0,
            _SHM_MAGIC,
            self.num_engines,
            self.chunk_tokens,
            1,
            -1.0,
            0,
        )
        if had_magic:
            self._mm[
                _SHM_INFLIGHT_OFFSET : _SHM_INFLIGHT_OFFSET + len(saved_inflight)
            ] = saved_inflight

    def reset(self) -> None:
        with self._lock():
            self._reset_unlocked()

    def close(self) -> None:
        if getattr(self, "_mm", None) is not None:
            self._mm.close()
            self._mm = None  # type: ignore[assignment]
        if getattr(self, "_fd", -1) >= 0:
            os.close(self._fd)
            self._fd = -1

    def sync_client_inflight(
        self, client_index: int, inflight_by_idx: list[int]
    ) -> None:
        """Write this APIServer's exact per-engine inflight counts into its exclusive row."""
        if (
            self._mm is None
            or not (0 <= client_index < _SHM_MAX_CLIENTS)
            or len(inflight_by_idx) != self.num_engines
        ):
            return
        off = _SHM_INFLIGHT_OFFSET + client_index * (_SHM_MAX_ENGINES * 4)
        struct.pack_into(self._row_fmt, self._mm, off, *inflight_by_idx)

    def read_global_inflight_unlocked(self, client_count: int) -> list[int]:
        n_clients = min(max(client_count, 1), _SHM_MAX_CLIENTS)
        totals = [0] * self.num_engines
        mm = self._mm
        fmt = self._row_fmt
        for c in range(n_clients):
            off = _SHM_INFLIGHT_OFFSET + c * (_SHM_MAX_ENGINES * 4)
            row = struct.unpack_from(fmt, mm, off)
            for e in range(self.num_engines):
                totals[e] += row[e]
        return totals

    def read_rr_cursor_unlocked(self) -> int:
        (rr,) = struct.unpack_from("<I", self._mm, 32)
        return int(rr) % max(self.num_engines, 1)

    def advance_rr_cursor_unlocked(self) -> None:
        (rr,) = struct.unpack_from("<I", self._mm, 32)
        struct.pack_into("<I", self._mm, 32, (int(rr) + 1) % max(self.num_engines, 1))

    def read_engine_counts_unlocked(self) -> list[int]:
        return list(
            struct.unpack_from(f"<{self.num_engines}q", self._mm, 64)
        )

    def write_engine_counts_unlocked(self, counts: list[int]) -> None:
        struct.pack_into(f"<{self.num_engines}q", self._mm, 64, *counts)

    def read_backlog_unlocked(self) -> list[float]:
        return list(
            struct.unpack_from(f"<{self.num_engines}d", self._mm, 576)
        )

    def write_backlog_unlocked(self, backlog: list[float]) -> None:
        struct.pack_into(f"<{self.num_engines}d", self._mm, 576, *backlog)

    def decay_backlog_unlocked(self, cur: float) -> list[float]:
        (last_t,) = struct.unpack_from("<d", self._mm, 24)
        backlog = self.read_backlog_unlocked()
        if last_t < 0.0:
            struct.pack_into("<d", self._mm, 24, cur)
            return backlog
        dt = max(0.0, cur - last_t)
        struct.pack_into("<d", self._mm, 24, cur)
        if dt <= 0.0:
            return backlog
        drained = dt * _PREFILL_DRAIN_TOKENS_PER_SEC * (16.0 / max(self.num_engines, 1))
        changed = False
        for i, prev in enumerate(backlog):
            if prev > 0.0:
                backlog[i] = max(0.0, prev - drained)
                changed = True
        if changed:
            self.write_backlog_unlocked(backlog)
        return backlog

    def lookup_mask_unlocked(self, digest: bytes) -> int:
        key = _shm_key(digest)
        set_idx = int.from_bytes(key[:4], "little") & _SHM_SET_MASK
        base = _SHM_HEADER_BYTES + set_idx * _SHM_SET_BYTES
        mm = self._mm
        unpack = _SHM_ENTRY_STRUCT.unpack_from
        for way in range(_SHM_WAYS):
            k_way, mask, _ = unpack(mm, base + way * _SHM_ENTRY_BYTES)
            if k_way == key:
                return mask
        return 0

    def cached_tokens_by_engine_unlocked(self, digests: list[bytes]) -> list[int]:
        cached = [0] * self.num_engines
        if not digests:
            return cached
        remaining = self.num_engines
        unit = self.chunk_tokens
        for pos in range(len(digests) - 1, -1, -1):
            mask = self.lookup_mask_unlocked(digests[pos])
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

    def record_unlocked(
        self,
        digests: list[bytes],
        eng_idx: int,
        prompt_len: int,
        cur: float,
    ) -> None:
        if prompt_len > 0:
            backlog = self.decay_backlog_unlocked(cur)
            matched = self.cached_tokens_by_engine_unlocked(digests)[eng_idx]
            uncached = max(0, prompt_len - matched)
            if uncached > 0:
                backlog[eng_idx] += float(uncached)
                self.write_backlog_unlocked(backlog)
        if not digests:
            return
        counts = self.read_engine_counts_unlocked()
        (seq,) = struct.unpack_from("<Q", self._mm, 16)
        bit = 1 << eng_idx
        mm = self._mm
        unpack = _SHM_ENTRY_STRUCT.unpack_from
        pack = _SHM_ENTRY_STRUCT.pack_into
        for d in digests:
            key = _shm_key(d)
            set_idx = int.from_bytes(key[:4], "little") & _SHM_SET_MASK
            base = _SHM_HEADER_BYTES + set_idx * _SHM_SET_BYTES
            found = False
            victim_way = 0
            victim_seq = (1 << 64) - 1
            victim_mask = 0
            for way in range(_SHM_WAYS):
                off = base + way * _SHM_ENTRY_BYTES
                k_way, mask, w_seq = unpack(mm, off)
                if k_way == key:
                    seq += 1
                    new_mask = mask | bit
                    if not (mask & bit):
                        counts[eng_idx] += 1
                    pack(mm, off, key, new_mask, seq)
                    found = True
                    break
                if k_way == _ZERO_KEY:
                    victim_way = way
                    victim_mask = 0
                    break
                if w_seq < victim_seq:
                    victim_seq = w_seq
                    victim_way = way
                    victim_mask = mask
            if not found:
                if victim_mask:
                    m = victim_mask
                    while m:
                        lsb = m & -m
                        e = lsb.bit_length() - 1
                        if e < self.num_engines and counts[e] > 0:
                            counts[e] -= 1
                        m ^= lsb
                seq += 1
                counts[eng_idx] += 1
                pack(mm, base + victim_way * _SHM_ENTRY_BYTES, key, bit, seq)
        struct.pack_into("<Q", mm, 16, seq)
        self.write_engine_counts_unlocked(counts)


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
        shm_path: str | None = None,
    ) -> None:
        self.num_engines = num_engines
        self.chunk_tokens = chunk_tokens
        self.max_entries = max_entries
        self._owners: OrderedDict[bytes, int] = OrderedDict()
        self._engine_counts: list[int] = [0] * num_engines
        self._engine_prefill_backlog: list[float] = [0.0] * num_engines
        self._last_decay_t: float | None = None
        self._shm_store: _SharedMmapPrefixStore | None = None
        if shm_path:
            try:
                self._shm_store = _SharedMmapPrefixStore(
                    shm_path, num_engines=num_engines, chunk_tokens=chunk_tokens
                )
            except OSError as exc:
                logger.warning(
                    "TPU DP prefix routing could not open shared store %s (%s); "
                    "falling back to process-local directory.",
                    shm_path,
                    exc,
                )

    def reset(self) -> None:
        """Clear all recorded chunk digests, counts, and prefill backlog."""
        self._owners.clear()
        self._engine_counts = [0] * self.num_engines
        self._engine_prefill_backlog = [0.0] * self.num_engines
        self._last_decay_t = None
        if self._shm_store is not None:
            self._shm_store.reset()

    def close(self) -> None:
        if self._shm_store is not None:
            self._shm_store.close()
            self._shm_store = None

    def sync_inflight(self, client_index: int, inflight_by_idx: list[int]) -> None:
        if self._shm_store is not None:
            self._shm_store.sync_client_inflight(client_index, inflight_by_idx)

    def _decay_prefill_backlog(self, now: float | None = None) -> None:
        cur = time.monotonic() if now is None else now
        if self._shm_store is not None:
            with self._shm_store._lock():
                self._engine_prefill_backlog = self._shm_store.decay_backlog_unlocked(
                    cur
                )
                self._engine_counts = self._shm_store.read_engine_counts_unlocked()
            return
        if self._last_decay_t is None:
            self._last_decay_t = cur
            return
        dt = max(0.0, cur - self._last_decay_t)
        self._last_decay_t = cur
        if dt <= 0.0:
            return
        drained_per_engine = (
            dt * _PREFILL_DRAIN_TOKENS_PER_SEC * (16.0 / max(self.num_engines, 1))
        )
        for idx in range(self.num_engines):
            prev = self._engine_prefill_backlog[idx]
            if prev > 0.0:
                self._engine_prefill_backlog[idx] = max(0.0, prev - drained_per_engine)

    def cached_tokens_by_engine(self, digests: list[bytes]) -> list[int]:
        """Return deepest matched prefix length (in tokens) for each DP engine."""
        if self._shm_store is not None:
            with self._shm_store._lock():
                return self._shm_store.cached_tokens_by_engine_unlocked(digests)
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
        client_index: int = 0,
        inflight_by_idx: list[int] | None = None,
    ) -> None:
        if not (0 <= eng_idx < self.num_engines):
            return
        if self._shm_store is not None:
            cur = time.monotonic() if now is None else now
            with self._shm_store._lock():
                if inflight_by_idx is not None:
                    self._shm_store.sync_client_inflight(client_index, inflight_by_idx)
                self._shm_store.record_unlocked(digests, eng_idx, prompt_len, cur)
                self._engine_counts = self._shm_store.read_engine_counts_unlocked()
                self._engine_prefill_backlog = self._shm_store.read_backlog_unlocked()
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
        client_index: int = 0,
        now: float | None = None,
    ) -> tuple[int, list[bytes]]:
        digests = _rolling_chunk_digests(prompt_token_ids, self.chunk_tokens)
        if self._shm_store is not None:
            cur = time.monotonic() if now is None else now
            with self._shm_store._lock():
                self._shm_store.sync_client_inflight(client_index, inflight_by_idx)
                eff_inflight = self._shm_store.read_global_inflight_unlocked(
                    client_count
                )
                start_idx = self._shm_store.read_rr_cursor_unlocked()
                self._shm_store.advance_rr_cursor_unlocked()
                backlog = self._shm_store.decay_backlog_unlocked(cur)
                counts = self._shm_store.read_engine_counts_unlocked()
                cached = self._shm_store.cached_tokens_by_engine_unlocked(digests)
                backlog_zeroed = False
                for idx in range(self.num_engines):
                    waiting, running, _ = lb_engines[idx]
                    if (
                        waiting == 0
                        and running == 0
                        and eff_inflight[idx] == 0
                        and backlog[idx] > 0.0
                    ):
                        backlog[idx] = 0.0
                        backlog_zeroed = True
                if backlog_zeroed:
                    self._shm_store.write_backlog_unlocked(backlog)
                self._engine_prefill_backlog = backlog
                self._engine_counts = counts
        else:
            self._decay_prefill_backlog(now)
            cached = self.cached_tokens_by_engine(digests)
            eff_inflight = [client_count * x for x in inflight_by_idx]
            start_idx = eng_start_index % self.num_engines
        prompt_len = len(prompt_token_ids)
        max_cached = max(cached) if cached else 0
        long_warm_session = (
            max_cached >= _STICKY_LONG_PREFIX_TOKENS
            and (prompt_len - max_cached) <= _WARM_FOLLOWUP_MAX_UNCACHED
        )
        uncached_scale = float(max(self.num_engines, 1)) if long_warm_session else 1.0
        avg_blocks = max(sum(self._engine_counts) / max(self.num_engines, 1), 1.0)

        best_idx = start_idx % self.num_engines
        best_cost = float(sys.maxsize)
        best_tie = self.num_engines

        for offset in range(self.num_engines):
            idx = (start_idx + offset) % self.num_engines
            waiting, running, kv_usage = lb_engines[idx]
            total_inflight = eff_inflight[idx]
            if waiting == 0 and running == 0 and total_inflight == 0:
                self._engine_prefill_backlog[idx] = 0.0
            unreported = max(0, total_inflight - (waiting + running))
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


def _default_shm_path(num_engines: int, chunk_tokens: int, client_count: int) -> str | None:
    explicit = os.getenv("TPU_DP_PREFIX_SHM_PATH", "").strip()
    if explicit:
        return None if explicit.lower() in ("0", "off", "false", "none") else explicit
    if client_count <= 1 or not os.path.isdir("/dev/shm"):
        return None
    return f"/dev/shm/vllm_tpu_dp_prefix_{os.getuid()}_{num_engines}_{chunk_tokens}.shm"


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
            engine = original(self, request)
            directory_bypass: _DPPrefixDirectory | None = getattr(
                self, "_tpu_dp_prefix_dir", None
            )
            if directory_bypass is not None:
                directory_bypass.sync_inflight(
                    int(getattr(self, "client_index", 0)),
                    [self.engine_inflight[eng] for eng in self.core_engines],
                )
            return engine

        directory: _DPPrefixDirectory | None = getattr(self, "_tpu_dp_prefix_dir", None)
        if directory is None or directory.num_engines != len(self.core_engines):
            raw_bs = int(self.vllm_config.cache_config.block_size or 256)
            eff_bs = 256 if raw_bs < 64 else raw_bs
            chunk_tokens = max(min(eff_bs, _DEFAULT_CHUNK_TOKENS), 64)
            shm_path = _default_shm_path(
                len(self.core_engines), chunk_tokens, int(getattr(self, "client_count", 1))
            )
            directory = _DPPrefixDirectory(
                num_engines=len(self.core_engines),
                chunk_tokens=chunk_tokens,
                shm_path=shm_path,
            )
            self._tpu_dp_prefix_dir = directory
            self._tpu_eng_to_idx = {eng: i for i, eng in enumerate(self.core_engines)}
            logger.info(
                "TPU DP cache-aware prefix routing active across %d DP engines "
                "(chunk_tokens=%d, client_count=%d, client_index=%d, shm=%s).",
                len(self.core_engines),
                chunk_tokens,
                int(getattr(self, "client_count", 1)),
                int(getattr(self, "client_index", 0)),
                shm_path or "disabled",
            )

        client_idx = int(getattr(self, "client_index", 0))
        inflight_by_idx = [self.engine_inflight[eng] for eng in self.core_engines]
        chosen_idx, digests = directory.select_engine(
            prompt_token_ids=request.prompt_token_ids,
            lb_engines=self.lb_engines,
            inflight_by_idx=inflight_by_idx,
            client_count=self.client_count,
            eng_start_index=self.eng_start_index,
            client_index=client_idx,
        )
        request.data_parallel_rank = chosen_idx
        try:
            engine = original(self, request)
        finally:
            request.data_parallel_rank = None
        actual_idx = self._tpu_eng_to_idx.get(engine, chosen_idx)
        # When process-local (shm disabled), update local lb_engines waiting count;
        # when shm is enabled, exact cross-process inflight counts in /dev/shm
        # already track burst arrivals between 100ms coordinator snapshots.
        if directory._shm_store is None and 0 <= actual_idx < len(self.lb_engines):
            entry = self.lb_engines[actual_idx]
            if isinstance(entry, list):
                entry[0] += self.client_count
            elif isinstance(entry, tuple) and len(entry) == 3:
                self.lb_engines[actual_idx] = (
                    entry[0] + self.client_count,
                    entry[1],
                    entry[2],
                )
        self.eng_start_index = (self.eng_start_index + 1) % len(self.core_engines)
        post_inflight = [self.engine_inflight[eng] for eng in self.core_engines]
        directory.record(
            digests,
            actual_idx,
            prompt_len=len(request.prompt_token_ids),
            client_index=client_idx,
            inflight_by_idx=post_inflight,
        )
        return engine

    setattr(get_core_engine_for_request, marker, True)
    cls.get_core_engine_for_request = get_core_engine_for_request

    orig_process_outputs = getattr(cls, "process_engine_outputs", None)
    if orig_process_outputs is not None and not getattr(
        orig_process_outputs, marker, False
    ):

        @wraps(orig_process_outputs)
        async def _wrapped_process_engine_outputs(self, outputs):
            await orig_process_outputs(self, outputs)
            if getattr(outputs, "finished_requests", None):
                directory: _DPPrefixDirectory | None = getattr(
                    self, "_tpu_dp_prefix_dir", None
                )
                if directory is not None and directory._shm_store is not None:
                    directory.sync_inflight(
                        int(getattr(self, "client_index", 0)),
                        [self.engine_inflight[eng] for eng in self.core_engines],
                    )

        setattr(_wrapped_process_engine_outputs, marker, True)
        cls.process_engine_outputs = staticmethod(_wrapped_process_engine_outputs)

    orig_reset_async = getattr(cls, "reset_prefix_cache_async", None)
    if orig_reset_async is not None and not getattr(orig_reset_async, marker, False):

        @wraps(orig_reset_async)
        async def reset_prefix_cache_async(self, *args, **kwargs):
            ok = await orig_reset_async(self, *args, **kwargs)
            directory: _DPPrefixDirectory | None = getattr(
                self, "_tpu_dp_prefix_dir", None
            )
            if directory is not None:
                directory.reset()
            else:
                raw_bs = int(self.vllm_config.cache_config.block_size or 256)
                eff_bs = 256 if raw_bs < 64 else raw_bs
                chunk_tokens = max(min(eff_bs, _DEFAULT_CHUNK_TOKENS), 64)
                shm_path = _default_shm_path(
                    len(self.core_engines),
                    chunk_tokens,
                    int(getattr(self, "client_count", 1)),
                )
                if shm_path and os.path.exists(shm_path):
                    tmp_store = _SharedMmapPrefixStore(
                        shm_path, len(self.core_engines), chunk_tokens
                    )
                    tmp_store.reset()
                    tmp_store.close()
            return ok

        setattr(reset_prefix_cache_async, marker, True)
        cls.reset_prefix_cache_async = reset_prefix_cache_async

    logger.info("Applied TPU patch: cache-aware DP prefix routing.")
