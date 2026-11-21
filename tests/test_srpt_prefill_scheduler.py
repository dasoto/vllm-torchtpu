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
"""SrptPrefillScheduler tests on a real scheduler, through schedule()."""

from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_torchtpu.compilation import tpu_compiler
from vllm_torchtpu.core.experimental.srpt_prefill_scheduler import SrptPrefillScheduler
from vllm_torchtpu.core.prefix_aware_scheduler import PrefixAwareScheduler

BLOCK_SIZE = 16
MODEL = "facebook/opt-125m"


@pytest.fixture(scope="module")
def shared_model_config():
    return ModelConfig(model=MODEL, dtype="float16", seed=42, max_model_len=2048)


def _build_real_scheduler(
    model_config: ModelConfig,
    *,
    scheduler_cls=SrptPrefillScheduler,
    num_blocks: int = 32,
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 64,
    long_prefill_token_threshold: int = 0,
    scheduler_reserve_full_isl: bool = False,
    policy: str = "fcfs",
):
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=2048,
            enable_chunked_prefill=True,
            long_prefill_token_threshold=long_prefill_token_threshold,
            scheduler_reserve_full_isl=scheduler_reserve_full_isl,
            is_encoder_decoder=False,
            watermark=0.0,
            async_scheduling=True,
            policy=policy,
        ),
        model_config=model_config,
        cache_config=CacheConfig(
            block_size=BLOCK_SIZE,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            enable_prefix_caching=True,
        ),
        parallel_config=ParallelConfig(),
        device_config=DeviceConfig(device="cpu"),
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer0"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=64,
                    dtype=torch.float16,
                ),
            )
        ],
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    register_all_kvcache_specs(vllm_config)
    scheduler = scheduler_cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=BLOCK_SIZE,
    )
    init_none_hash(sha256)
    hasher = get_request_block_hasher(BLOCK_SIZE, sha256)
    return scheduler, hasher


def _make_real_request(
    request_id: str,
    num_tokens: int,
    hasher,
    seed: int = 0,
    shared_prefix: list[int] | None = None,
) -> Request:
    params = SamplingParams(max_tokens=8, ignore_eos=True)
    params.update_from_generation_config({}, 2)
    prefix = list(shared_prefix) if shared_prefix is not None else []
    tail_len = num_tokens - len(prefix)
    prompt = prefix + [seed * 100_000 + i for i in range(tail_len)]
    return Request(request_id, prompt, params, None, block_hasher=hasher)


def _running_long_prefill(scheduler, hasher, num_tokens: int = 64) -> Request:
    """Admit `long` and run its first chunk so it is a running prefill."""
    req_long = _make_real_request("long", num_tokens, hasher, seed=1)
    scheduler.add_request(req_long)
    out = scheduler.schedule()
    assert set(out.num_scheduled_tokens) == {"long"}
    assert req_long.is_prefill_chunk is True
    return req_long


@pytest.mark.parametrize(
    ("scheduler_cls", "expected"),
    [
        # Admission order: `long` (48 left) takes its 16-token cap first.
        (AsyncScheduler, {"long": 16, "short": 8}),
        # SRPT order: `short` (40 left) goes first.
        (SrptPrefillScheduler, {"short": 16, "long": 8}),
    ],
)
def test_runs_running_prefill_chunks_fewest_remaining_first(
    shared_model_config, scheduler_cls, expected
):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        scheduler_cls=scheduler_cls,
        max_num_batched_tokens=24,
        long_prefill_token_threshold=16,
    )
    scheduler.add_request(_make_real_request("long", 64, hasher, seed=1))
    scheduler.add_request(_make_real_request("short", 48, hasher, seed=2))
    assert scheduler.schedule().num_scheduled_tokens == {"long": 16, "short": 8}

    assert scheduler.schedule().num_scheduled_tokens == expected


@pytest.mark.parametrize(
    ("scheduler_cls", "expected_step2_scheduled"),
    [
        # Upstream schedules running `long` first, which takes the whole
        # 16-token budget and leaves `short_wait` waiting.
        (PrefixAwareScheduler, {"long"}),
        # SRPT holds `long` (48 left > 16) so `short_wait` is admitted.
        (SrptPrefillScheduler, {"short_wait"}),
    ],
)
def test_holds_back_long_running_chunk_for_shorter_waiting_request(
    shared_model_config,
    scheduler_cls,
    expected_step2_scheduled,
):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        scheduler_cls=scheduler_cls,
        max_num_seqs=4,
        max_num_batched_tokens=16,
    )
    req_long = _running_long_prefill(scheduler, hasher)
    req_short_wait = _make_real_request("short_wait", 16, hasher, seed=2)
    scheduler.add_request(req_short_wait)

    out2 = scheduler.schedule()
    assert set(out2.num_scheduled_tokens) == expected_step2_scheduled
    assert scheduler.max_num_running_reqs == 4

    if scheduler_cls is SrptPrefillScheduler:
        # The held chunk is restored with its KV blocks.
        assert req_long in scheduler.running
        assert req_long.status == RequestStatus.RUNNING
        assert req_short_wait.is_prefill_chunk is False


def test_holds_for_next_waiting_request_without_reordering_waiting(
    shared_model_config,
):
    """SRPT holds for the request upstream admits next (queue order) and does
    not promote a warmer request behind it."""
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        max_num_batched_tokens=16,
    )
    assert isinstance(scheduler, AsyncScheduler)
    assert not isinstance(scheduler, PrefixAwareScheduler)
    # Prime 32 tokens (2 blocks) into the prefix cache.
    shared_prefix = [900_000 + i for i in range(32)]
    primer = _make_real_request(
        "primer", 32, hasher, seed=1, shared_prefix=shared_prefix
    )
    scheduler.add_request(primer)
    scheduler.schedule()
    scheduler.schedule()
    scheduler.finish_requests("primer", RequestStatus.FINISHED_ABORTED)
    scheduler.schedule()

    req_long = _make_real_request("long", 64, hasher, seed=10)
    scheduler.add_request(req_long)
    scheduler.schedule()
    assert req_long.is_prefill_chunk is True

    # `next` (32 uncached) arrives ahead of `warm` (48 tokens, 32 cached).
    scheduler.add_request(_make_real_request("next", 32, hasher, seed=2))
    scheduler.add_request(
        _make_real_request("warm", 48, hasher, seed=3, shared_prefix=shared_prefix)
    )

    out = scheduler.schedule()
    assert set(out.num_scheduled_tokens) == {"next"}
    assert [r.request_id for r in scheduler.waiting] == ["warm"]
    assert req_long in scheduler.running


def test_no_hold_when_waiting_request_is_longer(shared_model_config):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        max_num_batched_tokens=16,
    )
    _running_long_prefill(scheduler, hasher, num_tokens=32)  # 16 left
    scheduler.add_request(_make_real_request("longer", 48, hasher, seed=2))

    assert set(scheduler.schedule().num_scheduled_tokens) == {"long"}


def test_no_hold_at_max_num_seqs(shared_model_config):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        max_num_seqs=2,
        max_num_batched_tokens=32,
        long_prefill_token_threshold=16,
    )
    scheduler.add_request(_make_real_request("long", 64, hasher, seed=1))
    scheduler.add_request(_make_real_request("dec", 16, hasher, seed=2))
    assert set(scheduler.schedule().num_scheduled_tokens) == {"long", "dec"}
    # Both running slots are taken, so holding `long` cannot admit `short`.
    scheduler.add_request(_make_real_request("short", 16, hasher, seed=3))

    out = scheduler.schedule()
    assert "long" in out.num_scheduled_tokens
    assert "short" not in out.num_scheduled_tokens


@pytest.mark.parametrize(
    ("throttle_prefills", "policy"),
    [(True, "fcfs"), (False, "priority")],
)
def test_no_hold_on_throttled_step_or_priority_policy(
    shared_model_config, throttle_prefills, policy
):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        max_num_batched_tokens=16,
        policy=policy,
    )
    _running_long_prefill(scheduler, hasher)
    scheduler.add_request(_make_real_request("short_wait", 16, hasher, seed=2))

    out = scheduler.schedule(throttle_prefills=throttle_prefills)
    assert set(out.num_scheduled_tokens) == {"long"}


def test_holds_only_the_chunks_needed(shared_model_config):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        max_num_batched_tokens=48,
        long_prefill_token_threshold=32,
    )
    scheduler.add_request(_make_real_request("medium", 48, hasher, seed=1))
    scheduler.add_request(_make_real_request("long", 80, hasher, seed=2))
    assert scheduler.schedule().num_scheduled_tokens == {"medium": 32, "long": 16}

    # `medium` (16 left) plus `long` (32-token cap) fill the 48-token budget;
    # holding `long` alone leaves 32 for `short_wait`, so `medium` still runs.
    scheduler.add_request(_make_real_request("short_wait", 16, hasher, seed=3))
    out = scheduler.schedule()
    assert out.num_scheduled_tokens == {"medium": 16, "short_wait": 16}


def test_hold_never_exceeds_max_num_seqs(shared_model_config):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        max_num_seqs=4,
        max_num_batched_tokens=64,
        long_prefill_token_threshold=16,
    )
    for request_id, num_tokens, seed in (("dec_1", 16, 1), ("dec_2", 16, 2)):
        scheduler.add_request(_make_real_request(request_id, num_tokens, hasher, seed))
    scheduler.add_request(_make_real_request("long", 64, hasher, seed=3))
    assert set(scheduler.schedule().num_scheduled_tokens) == {"dec_1", "dec_2", "long"}

    # Holding `long` lowers the admission cap to 3 for the step, so only one of
    # two short requests is admitted and `long` restored makes 4, not 5.
    scheduler.add_request(_make_real_request("short_1", 16, hasher, seed=4))
    scheduler.add_request(_make_real_request("short_2", 16, hasher, seed=5))
    out = scheduler.schedule()
    assert "short_1" in out.num_scheduled_tokens
    assert "short_2" not in out.num_scheduled_tokens
    assert len(scheduler.running) == 4
    assert scheduler.max_num_running_reqs == 4
    assert scheduler.max_num_active_reqs == 4


def test_failed_hold_backs_off_exponentially_per_target(shared_model_config):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        max_num_seqs=4,
        max_num_batched_tokens=16,
    )
    _running_long_prefill(scheduler, hasher, num_tokens=160)
    # Deliverable stale output in flight: upstream skips this request in its
    # WAITING pass, so a hold for it admits nothing.
    req_stuck = _make_real_request("stuck", 16, hasher, seed=2)
    req_stuck.num_stale_output_tokens = 1
    req_stuck.drop_stale_output = False
    scheduler.add_request(req_stuck)

    scheduled = [set(scheduler.schedule().num_scheduled_tokens) for _ in range(5)]
    # Failed hold, 1 step off, failed hold, 2 steps off.
    assert scheduled == [set(), {"long"}, set(), {"long"}, {"long"}]

    req_stuck.num_stale_output_tokens = 0
    assert set(scheduler.schedule().num_scheduled_tokens) == {"stuck"}


@pytest.mark.parametrize(
    ("scheduler_cls", "expected_running", "expected_preempted"),
    [
        # Admission order leaves `dec` at running[-1], so it is preempted.
        (PrefixAwareScheduler, ["long", "short"], "dec"),
        # SRPT order puts `long` at running[-1], so it is preempted instead.
        (SrptPrefillScheduler, ["dec", "short"], "long"),
    ],
)
def test_kv_pressure_preempts_longest_prefill(
    shared_model_config,
    scheduler_cls,
    expected_running,
    expected_preempted,
):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        scheduler_cls=scheduler_cls,
        num_blocks=6,
        max_num_batched_tokens=48,
        long_prefill_token_threshold=BLOCK_SIZE,
    )
    reqs = {
        "long": _make_real_request("long", 64, hasher, seed=1),
        "short": _make_real_request("short", 32, hasher, seed=2),
        "dec": _make_real_request("dec", 16, hasher, seed=3),
    }
    for req in reqs.values():
        scheduler.add_request(req)
    assert set(scheduler.schedule().num_scheduled_tokens) == set(reqs)
    # An older long prefill ahead of newer requests, and no free KV blocks.
    scheduler.running[:] = [reqs["long"], reqs["short"], reqs["dec"]]
    pinned = _make_real_request("pinned", 16, hasher, seed=4)
    scheduler.kv_cache_manager.allocate_slots(pinned, 16)
    # A short waiting request must not trigger a hold under KV pressure.
    scheduler.add_request(_make_real_request("short_wait", 16, hasher, seed=5))

    out = scheduler.schedule()
    assert [r.request_id for r in scheduler.running] == expected_running
    assert set(out.num_scheduled_tokens) == set(expected_running)
    assert reqs[expected_preempted].status == RequestStatus.PREEMPTED


@pytest.mark.parametrize(
    ("scheduler_reserve_full_isl", "expected"), [(False, "wait_32"), (True, "long")]
)
def test_full_isl_reservation_checks_whole_prompt_before_holding(
    shared_model_config, scheduler_reserve_full_isl, expected
):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        num_blocks=16,
        max_num_batched_tokens=16,
        scheduler_reserve_full_isl=scheduler_reserve_full_isl,
    )
    _running_long_prefill(scheduler, hasher)
    # `long` (48 tokens left) reserves 3 more blocks; leave 4 free: room for one
    # 16-token chunk of `wait_32` (1 block) but not its whole prompt (2 blocks).
    pool = scheduler.kv_cache_manager.block_pool
    pin_tokens = (pool.get_num_free_blocks() - 4) * BLOCK_SIZE
    pinned = _make_real_request("pinned", pin_tokens, hasher, seed=6)
    scheduler.kv_cache_manager.allocate_slots(pinned, pin_tokens)
    assert pool.get_num_free_blocks() == 4
    scheduler.add_request(_make_real_request("wait_32", 32, hasher, seed=7))

    assert set(scheduler.schedule().num_scheduled_tokens) == {expected}


def test_hold_with_speculative_decoding_draft_slots(shared_model_config):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        max_num_batched_tokens=40,
    )
    _running_long_prefill(scheduler, hasher, num_tokens=96)
    # Scheduler.schedule() reads the drafting slots per step; a stand-in config
    # keeps the test independent of which platforms support a real drafter.
    scheduler.vllm_config.speculative_config = SimpleNamespace(
        max_num_new_slots_for_drafting=2
    )
    scheduler.add_request(_make_real_request("short_wait", 16, hasher, seed=2))

    out = scheduler.schedule()
    assert out.num_scheduled_tokens == {"short_wait": 16}
    # Upstream charges each scheduled request its 2 drafting slots.
    assert out.total_num_scheduled_tokens + 2 * len(out.num_scheduled_tokens) <= 40


def test_consecutive_holds_are_capped(shared_model_config):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        max_num_batched_tokens=16,
    )
    scheduler.max_consecutive_holds = 2
    req_long = _running_long_prefill(scheduler, hasher)

    for i in (1, 2):
        scheduler.add_request(_make_real_request(f"short_{i}", 16, hasher, seed=i + 1))
        out = scheduler.schedule()
        # `long` is held; earlier shorts may decode alongside.
        assert f"short_{i}" in out.num_scheduled_tokens
        assert "long" not in out.num_scheduled_tokens
    # Held twice in a row: `long` is not held again and runs its chunk.
    scheduler.add_request(_make_real_request("short_3", 16, hasher, seed=4))
    out = scheduler.schedule()
    assert "long" in out.num_scheduled_tokens
    assert req_long.num_computed_tokens > 16


def test_progress_is_measured_in_kv_cache_blocks(shared_model_config):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        max_num_batched_tokens=20,
    )
    scheduler.max_consecutive_holds = 2
    # Hybrid models use the LCM of all KV group block sizes here, which can
    # exceed a whole chunk; progress must not be measured against it.
    scheduler.block_size = 64 * BLOCK_SIZE
    req_long = _running_long_prefill(scheduler, hasher, num_tokens=96)

    for i in (1, 2):
        scheduler.add_request(_make_real_request(f"short_{i}", 16, hasher, seed=i + 1))
        assert "long" not in scheduler.schedule().num_scheduled_tokens
    # Stalled: runs a chunk of at least one KV cache block next to decodes.
    scheduler.add_request(_make_real_request("short_3", 16, hasher, seed=4))
    assert "long" in scheduler.schedule().num_scheduled_tokens
    computed = req_long.num_computed_tokens
    # Progress resets the count, so `long` can be held again.
    scheduler.add_request(_make_real_request("short_4", 16, hasher, seed=5))
    out = scheduler.schedule()
    assert "long" not in out.num_scheduled_tokens
    assert req_long.num_computed_tokens == computed


def test_prefill_crowded_out_by_shorter_running_prefill_is_promoted(
    shared_model_config,
):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        max_num_batched_tokens=16,
    )
    scheduler.max_consecutive_holds = 2
    req_long = _running_long_prefill(scheduler, hasher)
    scheduler.add_request(_make_real_request("mid", 40, hasher, seed=2))

    # Step 2: `long` is held so `mid` (40 < 48 tokens left) is admitted.
    assert set(scheduler.schedule().num_scheduled_tokens) == {"mid"}
    # Step 3: no hold, but `mid` (24 left) runs first and takes the budget.
    assert set(scheduler.schedule().num_scheduled_tokens) == {"mid"}
    # Two steps without a block of progress: `long` now runs ahead of `mid`.
    assert scheduler.schedule().num_scheduled_tokens == {"long": 16}
    assert req_long.num_computed_tokens == 32
    # Having advanced, `long` falls back behind the shorter `mid`.
    assert scheduler.schedule().num_scheduled_tokens == {"mid": 8, "long": 8}


def test_env_vars_are_read_and_skip_the_compile_cache(shared_model_config, monkeypatch):
    monkeypatch.setenv("VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS", "16")
    monkeypatch.setenv("VLLM_TPU_SRPT_ATTN_WORK_CAP", "25000000")
    monkeypatch.setenv("VLLM_TPU_SRPT_ATTN_TILE_TOKENS", "16")
    scheduler, _ = _build_real_scheduler(shared_model_config)
    assert scheduler.max_consecutive_holds == 16
    assert scheduler.attn_work_cap == 25_000_000
    assert scheduler.attn_tile_tokens == 16
    for name in (
        "VLLM_TPU_SRPT_MAX_CONSECUTIVE_HOLDS",
        "VLLM_TPU_SRPT_ATTN_WORK_CAP",
        "VLLM_TPU_SRPT_ATTN_TILE_TOKENS",
    ):
        assert name in tpu_compiler._TPU_COMPILE_ENV_IGNORED
        assert name not in tpu_compiler._tpu_compile_env_factors()


def _capped_long_prefill(shared_model_config, first_chunk: int = 128, **kwargs):
    """A scheduler with `long` (512 tokens) running its prefill after one
    uncapped chunk of `first_chunk` tokens, while another DP rank decodes."""
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        num_blocks=64,
        max_num_batched_tokens=first_chunk,
        **kwargs,
    )
    req_long = _running_long_prefill(scheduler, hasher, num_tokens=512)
    assert req_long.num_computed_tokens == first_chunk
    scheduler._other_rank_decoding = True
    return scheduler, hasher, req_long


@pytest.mark.parametrize(
    ("first_chunk", "cap", "tile", "expected"),
    [
        # After 128 computed tokens, 33 new ones cost 33 * (128 + 16.5) = 4768.5.
        (128, 4800, 32, 32),
        (128, 4800, 1, 33),
        # 50 fit (7650); rounding them up to 64 would cost 10240.
        (128, 7700, 32, 32),
        # 40 fit after 100 computed tokens (4800); the chunk ends at 128.
        (100, 4900, 32, 28),
    ],
)
def test_attn_work_cap_rounds_chunk_end_down_to_tile(
    shared_model_config, first_chunk, cap, tile, expected
):
    scheduler, _, _ = _capped_long_prefill(shared_model_config, first_chunk)
    scheduler.attn_work_cap = cap
    scheduler.attn_tile_tokens = tile
    assert scheduler.schedule().num_scheduled_tokens == {"long": expected}


@pytest.mark.parametrize(("first_chunk", "expected"), [(128, 32), (100, 28)])
def test_attn_work_cap_first_prefill_chunk_reaches_next_tile(
    shared_model_config, first_chunk, expected
):
    scheduler, _, _ = _capped_long_prefill(shared_model_config, first_chunk)
    # The context alone exceeds the cap; the chunk still advances to the next
    # multiple of the tile.
    scheduler.attn_work_cap = 1
    assert scheduler.schedule().num_scheduled_tokens == {"long": expected}


def test_attn_work_cap_skips_later_chunk_without_room_for_a_tile(
    shared_model_config,
):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        num_blocks=64,
        max_num_batched_tokens=256,
        long_prefill_token_threshold=64,
    )
    for name, seed in (("a", 1), ("b", 2)):
        scheduler.add_request(_make_real_request(name, 512, hasher, seed=seed))
    assert scheduler.schedule().num_scheduled_tokens == {"a": 64, "b": 64}
    # 32 tokens of `a` after 64 cost 32 * (64 + 16) = 2560; the 40 left fit
    # no tile of `b`.
    scheduler.attn_work_cap = 2600
    scheduler._other_rank_decoding = True
    assert scheduler.schedule().num_scheduled_tokens == {"a": 32}


def test_attn_work_cap_no_hold_once_target_reaches_next_tile(shared_model_config):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        num_blocks=128,
        max_num_batched_tokens=256,
        long_prefill_token_threshold=64,
    )
    # Cache one block, so `warm` starts at 16, off the 32-token tile grid.
    shared_prefix = [900_000 + i for i in range(BLOCK_SIZE)]
    scheduler.add_request(
        _make_real_request("primer", BLOCK_SIZE, hasher, shared_prefix=shared_prefix)
    )
    scheduler.schedule()
    scheduler.finish_requests("primer", RequestStatus.FINISHED_ABORTED)
    for name, seed in (("a", 1), ("b", 2)):
        scheduler.add_request(_make_real_request(name, 512, hasher, seed=seed))
    assert scheduler.schedule().num_scheduled_tokens == {"a": 64, "b": 64}
    # 32 tokens of `a` after 64 cost 2560. The 640 left fit no tile of `b`,
    # but take `warm` (32 tokens to go) from 16 to the tile at 32, so `a` is
    # not held for it.
    scheduler.attn_work_cap = 3200
    scheduler._other_rank_decoding = True
    scheduler.add_request(
        _make_real_request("warm", 48, hasher, seed=3, shared_prefix=shared_prefix)
    )
    assert scheduler.schedule().num_scheduled_tokens == {"a": 32, "warm": 16}


def test_attn_capped_chunk_counts_as_progress(shared_model_config):
    scheduler, hasher, _ = _capped_long_prefill(shared_model_config)
    scheduler.max_consecutive_holds = 2
    # Progress is one KV cache block; make it larger than a capped chunk.
    scheduler.cache_config.block_size = 64
    scheduler.attn_work_cap = 1
    for _ in range(3):
        assert scheduler.schedule().num_scheduled_tokens == {"long": 32}
    # `long` is not stalled, so it can still be held for a shorter request.
    scheduler.add_request(_make_real_request("short", 16, hasher, seed=2))
    assert scheduler.schedule().num_scheduled_tokens == {"short": 16}


def test_attn_work_cap_applies_only_while_some_rank_decodes(
    shared_model_config, monkeypatch
):
    scheduler, _, _ = _capped_long_prefill(shared_model_config)
    scheduler.attn_work_cap = 1
    # Only the decode report is under test, not upstream's output handling.
    monkeypatch.setattr(AsyncScheduler, "update_from_output", lambda *args: {})
    # No rank decodes: the chunk takes the whole budget.
    scheduler.update_from_output(None, SimpleNamespace(other_rank_decoding=False))
    assert scheduler.schedule().num_scheduled_tokens == {"long": 128}
    # Another rank decoded in the last reported step: the chunk is capped.
    scheduler.update_from_output(None, SimpleNamespace(other_rank_decoding=True))
    assert scheduler.schedule().num_scheduled_tokens == {"long": 32}
    # The shared empty output of a step without tokens carries no report.
    scheduler.update_from_output(None, SimpleNamespace())
    assert scheduler.schedule().num_scheduled_tokens == {"long": 32}


def test_attn_work_cap_applies_while_this_rank_decodes(shared_model_config):
    scheduler, hasher, _ = _capped_long_prefill(shared_model_config)
    scheduler._other_rank_decoding = False
    scheduler.attn_work_cap = 1
    # `long` is held so `dec` is admitted; `dec` then decodes.
    scheduler.add_request(_make_real_request("dec", 16, hasher, seed=2))
    assert scheduler.schedule().num_scheduled_tokens == {"dec": 16}
    # Uncapped, `long` would take the 127 tokens the decode leaves.
    assert scheduler.schedule().num_scheduled_tokens == {"dec": 1, "long": 32}


def test_attn_work_cap_holds_chunk_with_most_attention_work(shared_model_config):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        num_blocks=128,
        max_num_batched_tokens=256,
        long_prefill_token_threshold=64,
    )
    scheduler.add_request(_make_real_request("fresh", 512, hasher, seed=1))
    scheduler.max_num_scheduled_tokens = 16
    assert scheduler.schedule().num_scheduled_tokens == {"fresh": 16}
    scheduler.max_num_scheduled_tokens = 64
    scheduler.add_request(_make_real_request("deep", 448, hasher, seed=2))
    for _ in range(4):
        assert scheduler.schedule().num_scheduled_tokens == {"deep": 64}
    scheduler.max_num_scheduled_tokens = 256

    # `deep` (192 left at context 256) takes 64 tokens = 18432 work; `fresh`
    # (496 left at context 16) is capped to 16 tokens = 384 work. Token budget
    # is left (176 >= 64), so holding `deep` rather than the longer-remaining
    # `fresh` frees enough attention work in one hold for both `fresh` and `wait`.
    scheduler.attn_work_cap = 18900
    scheduler._other_rank_decoding = True
    scheduler.add_request(_make_real_request("wait", 64, hasher, seed=3))
    assert scheduler.schedule().num_scheduled_tokens == {"fresh": 64, "wait": 64}


def test_cadence_no_escape_keeps_defer_prefills_with_waiting_backlog(
    shared_model_config,
):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        num_blocks=128,
        max_num_seqs=8,
        max_num_batched_tokens=64,
    )
    scheduler.cadence_no_escape = True
    # Admit `dec` (16 tokens) so it becomes a decode request on step 2.
    scheduler.add_request(_make_real_request("dec", 16, hasher, seed=1))
    assert scheduler.schedule(throttle_prefills=False).num_scheduled_tokens == {
        "dec": 16
    }
    # Add two long prefills (`p1` and `p2`, 128 tokens each). Release step fills
    # the 64-token budget (`dec`: 1 + `p1`: 63) and leaves `p2` in `waiting`.
    scheduler.add_request(_make_real_request("p1", 128, hasher, seed=2))
    scheduler.add_request(_make_real_request("p2", 128, hasher, seed=3))
    out_release = scheduler.schedule(throttle_prefills=False)
    assert out_release.num_scheduled_tokens == {"dec": 1, "p1": 63}
    assert scheduler.prefill_capacity_bound is False
    # Next step has `throttle_prefills=True`: with `cadence_no_escape`, neither
    # `p1` (in-progress prefill chunk) nor `p2` (waiting) runs; only `dec` runs.
    out_throttled = scheduler.schedule(throttle_prefills=True)
    assert out_throttled.num_scheduled_tokens == {"dec": 1}


def test_cadence_no_escape_holds_prefills_when_other_rank_decoding(
    shared_model_config, monkeypatch
):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        num_blocks=128,
        max_num_seqs=8,
        max_num_batched_tokens=64,
    )
    scheduler.cadence_no_escape = True
    monkeypatch.setattr(AsyncScheduler, "update_from_output", lambda *args: {})
    scheduler.add_request(_make_real_request("p1", 128, hasher, seed=10))
    scheduler.add_request(_make_real_request("p2", 128, hasher, seed=11))
    # Step 1 (release): `p1` runs 64 tokens (`p1` still prefilling, `p2` waiting,
    # 0 local decode requests).
    assert scheduler.schedule(throttle_prefills=False).num_scheduled_tokens == {
        "p1": 64
    }
    # When another DP rank is decoding on a throttled step, hold `p1` and pause `p2`.
    scheduler.update_from_output(None, SimpleNamespace(other_rank_decoding=True))
    assert scheduler.schedule(throttle_prefills=True).num_scheduled_tokens == {}
    # When NO DP rank is decoding, full prefill runs unthrottled.
    scheduler.update_from_output(None, SimpleNamespace(other_rank_decoding=False))
    assert scheduler.schedule(throttle_prefills=True).num_scheduled_tokens == {
        "p1": 64
    }


def test_promote_warm_waiting_prioritizes_prefix_hit(shared_model_config):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        num_blocks=128,
        max_num_seqs=8,
        max_num_batched_tokens=64,
    )
    scheduler.promote_warm_waiting = True
    # Populate prefix cache with a 48-token conversation prefix (seed=99).
    seed_req = _make_real_request("seed_turn", 48, hasher, seed=99)
    scheduler.add_request(seed_req)
    assert scheduler.schedule().num_scheduled_tokens == {"seed_turn": 48}
    scheduler.schedule()
    scheduler.finish_requests("seed_turn", RequestStatus.FINISHED_ABORTED)
    scheduler.schedule()

    # Enqueue a cold 64-token Turn-0 request first, followed by a warm 64-token
    # follow-up turn sharing the 48-token cached prefix (`16` uncached tokens).
    cold_req = _make_real_request("cold_t0", 64, hasher, seed=55)
    warm_req = _make_real_request(
        "warm_t1",
        64,
        hasher,
        seed=100,
        shared_prefix=list(seed_req.prompt_token_ids),
    )
    scheduler.add_request(cold_req)
    scheduler.add_request(warm_req)
    out = scheduler.schedule()
    assert list(out.num_scheduled_tokens.keys())[0] == "warm_t1"
    assert out.num_scheduled_tokens["warm_t1"] == 16


def test_promote_warm_waiting_bypasses_stalled_kv_fetch(shared_model_config):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        num_blocks=128,
        max_num_seqs=8,
        max_num_batched_tokens=64,
    )
    scheduler.promote_warm_waiting = True
    stalled_req = _make_real_request("stalled_fetch", 64, hasher, seed=201)
    cold_req = _make_real_request("cold_ready", 64, hasher, seed=202)
    scheduler.add_request(stalled_req)
    scheduler.add_request(cold_req)
    scheduler._kv_fetch_stages = {stalled_req: "waiting_to_start"}
    scheduler._maybe_promote_warm_waiting()
    assert list(scheduler.waiting)[0].request_id == "cold_ready"


def test_promote_warm_waiting_promotes_multiple_warm_requests_in_order(
    shared_model_config,
):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        num_blocks=128,
        max_num_seqs=8,
        max_num_batched_tokens=64,
    )
    scheduler.promote_warm_waiting = True
    # Populate prefix cache with two conversation prefixes (48 and 32 tokens).
    seed_a = _make_real_request("seed_a", 48, hasher, seed=301)
    seed_b = _make_real_request("seed_b", 32, hasher, seed=302)
    scheduler.add_request(seed_a)
    assert scheduler.schedule().num_scheduled_tokens == {"seed_a": 48}
    scheduler.schedule()
    scheduler.finish_requests("seed_a", RequestStatus.FINISHED_ABORTED)
    scheduler.add_request(seed_b)
    assert scheduler.schedule().num_scheduled_tokens == {"seed_b": 32}
    scheduler.schedule()
    scheduler.finish_requests("seed_b", RequestStatus.FINISHED_ABORTED)
    scheduler.schedule()

    # Enqueue two long cold requests ahead of two short warm follow-ups:
    # warm_b has 32 uncached tokens, warm_a has 16 uncached tokens; both fit
    # together in a single 64-token step (`16 + 32 = 48 <= 64`) once promoted!
    scheduler.add_request(_make_real_request("cold_1", 128, hasher, seed=310))
    scheduler.add_request(_make_real_request("cold_2", 128, hasher, seed=311))
    scheduler.add_request(
        _make_real_request(
            "warm_b", 64, hasher, seed=312, shared_prefix=list(seed_b.prompt_token_ids)
        )
    )
    scheduler.add_request(
        _make_real_request(
            "warm_a", 64, hasher, seed=313, shared_prefix=list(seed_a.prompt_token_ids)
        )
    )
    out = scheduler.schedule()
    assert list(out.num_scheduled_tokens.keys())[:2] == ["warm_a", "warm_b"]
    assert out.num_scheduled_tokens["warm_a"] == 16
    assert out.num_scheduled_tokens["warm_b"] == 32


def test_estimated_prefix_hit_uses_external_connector_hashes_after_hbm_reset(
    shared_model_config,
):
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        num_blocks=128,
        max_num_seqs=8,
        max_num_batched_tokens=64,
    )
    # Attach a stub external connector with 32-token chunk alignment.
    stub_conn = SimpleNamespace(
        _scheduler=SimpleNamespace(config=SimpleNamespace(alignment_tokens=32)),
        supports_divergent_local_hybrid_hits=False,
        requires_kv_delivery=False,
        on_new_request=lambda req: None,
        get_num_new_matched_tokens=lambda req, n: (0, False),
        update_state_after_alloc=lambda req, blocks, n: None,
        build_connector_meta=lambda out: None,
        request_finished=lambda req, block_ids: (False, None),
        has_connector_metadata=lambda: False,
        has_pending_block_frees=lambda: False,
        has_pending_push_work=lambda: False,
        reset_cache=lambda: True,
        get_external_cache_hit_sources=lambda req_ids: {},
    )
    scheduler.connector = stub_conn
    seed_req = _make_real_request("seed_ext", 48, hasher, seed=401)
    scheduler.add_request(seed_req)
    scheduler.schedule()
    scheduler.connector = None
    scheduler.finish_requests("seed_ext", RequestStatus.FINISHED_ABORTED)
    scheduler.schedule()

    # Flush local HBM prefix cache without resetting external connector tier.
    assert scheduler.reset_prefix_cache(reset_connector=False) is True
    scheduler.connector = stub_conn
    followup = _make_real_request(
        "followup_ext",
        64,
        hasher,
        seed=402,
        shared_prefix=list(seed_req.prompt_token_ids),
    )
    # 48 matched hash tokens aligned down to 32-token connector chunk -> 32 tokens.
    assert scheduler._estimated_prefix_hit_tokens(followup) == 32
    scheduler.connector = None
    assert scheduler.reset_prefix_cache(reset_connector=True) is True
    scheduler.connector = stub_conn
    assert scheduler._estimated_prefix_hit_tokens(followup) == 0


def test_tpu_runner_any_decode_enabled_by_cadence_no_escape(monkeypatch):
    import numpy as np

    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    monkeypatch.setenv("VLLM_TPU_SRPT_ATTN_WORK_CAP", "0")
    monkeypatch.setenv("VLLM_TPU_PREFILL_CADENCE_NO_ESCAPE", "1")
    runner = TPUModelRunner.__new__(TPUModelRunner)
    runner.input_batch = SimpleNamespace(
        req_id_to_index={"dec_r": 0, "pref_r": 1},
        num_computed_tokens_cpu=np.array([64, 32], dtype=np.int32),
        num_prompt_tokens=np.array([64, 128], dtype=np.int32),
    )
    assert not runner._any_decode(SimpleNamespace(num_scheduled_tokens={}))
    assert not runner._any_decode(SimpleNamespace(num_scheduled_tokens={"pref_r": 32}))
    assert runner._any_decode(SimpleNamespace(num_scheduled_tokens={"dec_r": 1}))


def test_progress_when_cache_block_size_exceeds_chunk_budget(shared_model_config):
    """When DCP / hybrid unified pooling scales cache_config.block_size above
    max_num_batched_tokens, chunks getting their full budget must not be
    falsely marked stalled after max_consecutive_holds steps."""
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        num_blocks=64,
        max_num_seqs=8,
        max_num_batched_tokens=16,
    )
    scheduler.max_consecutive_holds = 2
    scheduler.cache_config.block_size = 64  # > max_num_batched_tokens (16)

    req_long = _running_long_prefill(scheduler, hasher, num_tokens=160)
    # Advance two full 16-token chunks: must NOT accumulate _stalled_steps.
    for _ in range(2):
        out = scheduler.schedule()
        assert out.num_scheduled_tokens.get("long") == 16
    assert scheduler._stalled_steps.get("long", 0) == 0

    # A short waiting arrival can still hold `long` because `long` is not stalled.
    scheduler.add_request(_make_real_request("short_1", 16, hasher, seed=11))
    out = scheduler.schedule()
    assert "short_1" in out.num_scheduled_tokens
    assert "long" not in out.num_scheduled_tokens

    # After 2 consecutive holds (`max_consecutive_holds=2`), anti-starvation kicks in.
    scheduler.add_request(_make_real_request("short_2", 16, hasher, seed=12))
    out = scheduler.schedule()
    assert "short_2" in out.num_scheduled_tokens
    assert "long" not in out.num_scheduled_tokens
    assert scheduler._is_stalled(req_long)

    scheduler.add_request(_make_real_request("short_3", 16, hasher, seed=13))
    out = scheduler.schedule()
    assert "long" in out.num_scheduled_tokens
    assert not scheduler._is_stalled(req_long)


def test_connector_scheduler_partial_tail_alignment_and_hbm_headroom_promotion(
    shared_model_config, monkeypatch
):
    """Verify connector_scheduler partial-tail hash alignment, hard-starvation
    tier ordering, and non-head-of-line HBM admission under full-ISL reservation."""
    monkeypatch.setenv("VLLM_TPU_SRPT_PROMOTE_WARM_WAITING", "1")
    scheduler, hasher = _build_real_scheduler(
        shared_model_config,
        num_blocks=10,  # 9 usable KV blocks (block 0 is null)
        max_num_seqs=8,
        max_num_batched_tokens=64,
        scheduler_reserve_full_isl=True,
    )
    stub_conn = SimpleNamespace(
        connector_scheduler=SimpleNamespace(
            config=SimpleNamespace(
                supports_partial_tail=True,
                tokens_per_hash=BLOCK_SIZE,
                alignment_tokens=4 * BLOCK_SIZE,
            )
        ),
        supports_divergent_local_hybrid_hits=False,
        requires_kv_delivery=False,
        on_new_request=lambda req: None,
        get_num_new_matched_tokens=lambda req, n: (0, False),
        update_state_after_alloc=lambda req, blocks, n: None,
        build_connector_meta=lambda out: None,
        request_finished=lambda req, block_ids: (False, None),
        has_connector_metadata=lambda: False,
        has_pending_block_frees=lambda: False,
        has_pending_push_work=lambda: False,
        reset_cache=lambda: True,
        get_external_cache_hit_sources=lambda req_ids: {},
    )
    scheduler.connector = stub_conn
    seed_req = _make_real_request("seed_pt", 3 * BLOCK_SIZE, hasher, seed=501)
    scheduler.add_request(seed_req)
    scheduler.schedule()
    scheduler.connector = None
    scheduler.finish_requests("seed_pt", RequestStatus.FINISHED_ABORTED)
    scheduler.schedule()
    assert scheduler.reset_prefix_cache(reset_connector=False) is True
    scheduler.connector = stub_conn

    followup = _make_real_request(
        "followup_pt",
        4 * BLOCK_SIZE,
        hasher,
        seed=502,
        shared_prefix=list(seed_req.prompt_token_ids),
    )
    # With supports_partial_tail=True, 3 * BLOCK_SIZE (48) stays 48 instead of 0.
    assert scheduler._estimated_prefix_hit_tokens(followup) == 3 * BLOCK_SIZE
    scheduler.connector = None

    # Now occupy 5 of the 9 usable KV blocks with an active request, leaving 4 free.
    active = _make_real_request("active", 5 * BLOCK_SIZE, hasher, seed=503)
    scheduler.add_request(active)
    out = scheduler.schedule()
    assert "active" in out.num_scheduled_tokens

    # Enqueue an oversize request (needs 6 blocks > 4 free) ahead of a fitting
    # request (needs 3 blocks <= 4 free). Without HBM-headroom promotion,
    # waiting[0] (`oversize`) would fail allocate_slots and `break`, admitting nothing.
    oversize = _make_real_request("oversize", 6 * BLOCK_SIZE, hasher, seed=504)
    fitting = _make_real_request("fitting", 3 * BLOCK_SIZE, hasher, seed=505)
    scheduler.add_request(oversize)
    scheduler.add_request(fitting)

    out2 = scheduler.schedule()
    assert "fitting" in out2.num_scheduled_tokens
    assert "oversize" not in out2.num_scheduled_tokens

    # Also verify hard-starvation tier (-1) orders a >600s aged request ahead of
    # a newly arrived short request when both fit in free HBM blocks.
    scheduler.finish_requests(
        ["active", "fitting", "oversize"], RequestStatus.FINISHED_ABORTED
    )
    scheduler.schedule()
    aged_long = _make_real_request("aged_long", 4 * BLOCK_SIZE, hasher, seed=506)
    aged_long.arrival_time -= 700.0
    fresh_short = _make_real_request("fresh_short", BLOCK_SIZE, hasher, seed=507)
    scheduler.add_request(fresh_short)
    scheduler.add_request(aged_long)
    scheduler._maybe_promote_warm_waiting()
    assert list(scheduler.waiting)[0].request_id == "aged_long"


