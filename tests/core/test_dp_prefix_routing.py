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
"""Unit tests for cache-aware DP prefix routing."""

import asyncio
import os
import tempfile
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

from vllm_torchtpu.core.dp_prefix_routing import (
    _DPPrefixDirectory,
    _rolling_chunk_digests,
    patch_dp_prefix_routing,
)


class TestDPPrefixDirectory(unittest.TestCase):
    def test_rolling_digests_chain_sensitivity(self):
        a = list(range(1024))
        b = list(range(512)) + [999999] + list(range(513, 1024))
        c = [999999] + list(range(1, 1024))
        da = _rolling_chunk_digests(a, 512)
        db = _rolling_chunk_digests(b, 512)
        dc = _rolling_chunk_digests(c, 512)
        self.assertEqual(len(da), 2)
        self.assertEqual(da[0], db[0])
        self.assertNotEqual(da[1], db[1])
        self.assertNotEqual(da[0], dc[0])
        # Second chunk depends on first chunk via chained person bytes
        self.assertNotEqual(da[1], dc[1])

    def test_shared_system_prompt_balances_cold_and_sticks_multiturn(self):
        directory = _DPPrefixDirectory(num_engines=4, chunk_tokens=512)
        system = [100 + i for i in range(7000)]
        homes: dict[int, int] = {}
        turn0_per_engine = [0] * 4

        for s in range(40):
            prompt = system + [50_000 + s * 1000 + i for i in range(20_000)]
            lb = [(0, 0, c / 20_000.0) for c in directory._engine_counts]
            eng, digests = directory.select_engine(
                prompt_token_ids=prompt,
                lb_engines=lb,
                inflight_by_idx=[0, 0, 0, 0],
                client_count=1,
                eng_start_index=s % 4,
            )
            directory.record(digests, eng)
            homes[s] = eng
            turn0_per_engine[eng] += 1

        # All 4 DP engines receive balanced cold sessions despite 7k system prompt.
        for count in turn0_per_engine:
            self.assertGreaterEqual(count, 8)
            self.assertLessEqual(count, 12)

        # Follow-up turns route 100% to each session's home DP engine.
        for s in range(40):
            followup = (
                system
                + [50_000 + s * 1000 + i for i in range(20_000)]
                + [900_000 + s * 100 + j for j in range(1500)]
            )
            lb = [(0, 0, c / 20_000.0) for c in directory._engine_counts]
            eng, digests = directory.select_engine(
                prompt_token_ids=followup,
                lb_engines=lb,
                inflight_by_idx=[0, 0, 0, 0],
                client_count=1,
                eng_start_index=(s + 1) % 4,
            )
            self.assertEqual(eng, homes[s])
            directory.record(digests, eng)

    def test_overload_spills_to_idle_engine(self):
        directory = _DPPrefixDirectory(num_engines=4, chunk_tokens=512)
        prompt = [1000 + i for i in range(10_240)]
        _, digests = directory.select_engine(
            prompt_token_ids=prompt,
            lb_engines=[(0, 0, 0.0)] * 4,
            inflight_by_idx=[0] * 4,
            client_count=1,
            eng_start_index=0,
        )
        directory.record(digests, 0)

        # Engine 0 has 10k cached tokens, but 5 waiting + 8 running requests.
        lb = [(5, 8, 0.8), (0, 0, 0.1), (0, 0, 0.1), (0, 0, 0.1)]
        eng, _ = directory.select_engine(
            prompt_token_ids=prompt + [42] * 512,
            lb_engines=lb,
            inflight_by_idx=[13, 0, 0, 0],
            client_count=1,
            eng_start_index=0,
        )
        self.assertNotEqual(eng, 0)

    def test_long_warm_multiturn_sticks_under_queue_load(self):
        directory = _DPPrefixDirectory(num_engines=8, chunk_tokens=512)
        prompt = [2000 + i for i in range(55_296)]
        _, digests = directory.select_engine(
            prompt_token_ids=prompt,
            lb_engines=[(0, 0, 0.0)] * 8,
            inflight_by_idx=[0] * 8,
            client_count=1,
            eng_start_index=0,
        )
        directory.record(digests, 3)

        # Engine 3 owns the 55K conversation prefix and has 8 waiting + 12 running,
        # while other DP engines are idle. Under EP lockstep + warm waiting
        # promotion, Turn 1 (+512 tokens) must stick to Engine 3.
        lb = [(0, 0, 0.1)] * 8
        lb[3] = (8, 12, 0.85)
        inflight = [0] * 8
        inflight[3] = 20
        eng, _ = directory.select_engine(
            prompt_token_ids=prompt + [77] * 512,
            lb_engines=lb,
            inflight_by_idx=inflight,
            client_count=1,
            eng_start_index=0,
        )
        self.assertEqual(eng, 3)

    def test_12k_warm_multiturn_sticks_on_dp8_and_dp16(self):
        for dp_size in (8, 16):
            directory = _DPPrefixDirectory(num_engines=dp_size, chunk_tokens=512)
            prompt = [3000 + i for i in range(12_288)]
            _, digests = directory.select_engine(
                prompt_token_ids=prompt,
                lb_engines=[(0, 0, 0.0)] * dp_size,
                inflight_by_idx=[0] * dp_size,
                client_count=1,
                eng_start_index=0,
            )
            home = dp_size - 1
            directory.record(digests, home)

            lb = [(0, 0, 0.2)] * dp_size
            lb[home] = (6, 12, 0.65)
            inflight = [0] * dp_size
            inflight[home] = 18
            eng, _ = directory.select_engine(
                prompt_token_ids=prompt + [88] * 2_560,
                lb_engines=lb,
                inflight_by_idx=inflight,
                client_count=1,
                eng_start_index=0,
            )
            self.assertEqual(eng, home)

    def test_45k_uncached_warm_multiturn_sticks_on_dp8(self):
        directory = _DPPrefixDirectory(num_engines=8, chunk_tokens=512)
        prompt = [4000 + i for i in range(81_920)]
        _, digests = directory.select_engine(
            prompt_token_ids=prompt,
            lb_engines=[(0, 0, 0.0)] * 8,
            inflight_by_idx=[0] * 8,
            client_count=1,
            eng_start_index=0,
        )
        directory.record(digests, 5)

        lb = [(0, 0, 0.15)] * 8
        lb[5] = (6, 14, 0.75)
        inflight = [0] * 8
        inflight[5] = 20
        eng, _ = directory.select_engine(
            prompt_token_ids=prompt + [99] * 45_056,
            lb_engines=lb,
            inflight_by_idx=inflight,
            client_count=1,
            eng_start_index=0,
        )
        self.assertEqual(eng, 5)

    def test_patch_dp_prefix_routing_respects_explicit_rank(self):
        class FakeClient:
            def __init__(self):
                self.vllm_config = SimpleNamespace(
                    cache_config=SimpleNamespace(
                        enable_prefix_caching=True, block_size=512
                    )
                )
                self.core_engines = ["e0", "e1"]
                self.lb_engines = [(0, 0, 0.0), (0, 0, 0.0)]
                self.engine_inflight = {"e0": 0, "e1": 0}
                self.client_count = 1
                self.eng_start_index = 0
                self.reqs_in_flight = {}

            def get_core_engine_for_request(self, request):
                idx = (
                    request.data_parallel_rank
                    if request.data_parallel_rank is not None
                    else 0
                )
                return self.core_engines[idx]

        fake_mod = SimpleNamespace(DPLBAsyncMPClient=FakeClient)
        fake_eng = SimpleNamespace(core_client=fake_mod)
        fake_v1 = SimpleNamespace(engine=fake_eng)
        with patch.dict(
            "sys.modules",
            {
                "vllm": SimpleNamespace(v1=fake_v1),
                "vllm.v1": fake_v1,
                "vllm.v1.engine": fake_eng,
                "vllm.v1.engine.core_client": fake_mod,
            },
        ):
            patch_dp_prefix_routing()

        client = FakeClient()
        req_pinned = SimpleNamespace(
            data_parallel_rank=1,
            resumable=False,
            request_id="r1",
            prompt_token_ids=list(range(2048)),
        )
        self.assertEqual(client.get_core_engine_for_request(req_pinned), "e1")
        self.assertEqual(req_pinned.data_parallel_rank, 1)

        # Unpinned request goes through cache-aware routing, increments local
        # lb_engines waiting count, and rotates eng_start_index.
        req_auto = SimpleNamespace(
            data_parallel_rank=None,
            resumable=False,
            request_id="r2",
            prompt_token_ids=list(range(5000, 7048)),
        )
        chosen = client.get_core_engine_for_request(req_auto)
        self.assertEqual(chosen, "e0")
        self.assertIsNone(req_auto.data_parallel_rank)
        self.assertEqual(client.lb_engines[0][0], 1)
        self.assertEqual(client.eng_start_index, 1)

    def test_cold_prefill_backlog_spreads_80k_arrivals_while_warm_sticks(self):
        directory = _DPPrefixDirectory(num_engines=4, chunk_tokens=512)
        cold_0 = [10_000 + i for i in range(81_920)]
        eng0, dig0 = directory.select_engine(
            prompt_token_ids=cold_0,
            lb_engines=[(0, 0, 0.1)] * 4,
            inflight_by_idx=[0] * 4,
            client_count=1,
            eng_start_index=0,
            now=100.0,
        )
        self.assertEqual(eng0, 0)
        directory.record(dig0, eng0, prompt_len=len(cold_0), now=100.0)

        # Even if engine 0 now reports (waiting=0, running=1) while engines 1..3
        # have 3 fast decodes (running=3), engine 0 still has ~81K uncached
        # prefill tokens in flight and must not receive a second 80K cold prompt.
        lb = [(0, 1, 0.1), (0, 3, 0.1), (0, 3, 0.1), (0, 3, 0.1)]
        cold_1 = [200_000 + i for i in range(81_920)]
        eng1, _ = directory.select_engine(
            prompt_token_ids=cold_1,
            lb_engines=lb,
            inflight_by_idx=[1, 3, 3, 3],
            client_count=1,
            eng_start_index=0,
            now=101.0,
        )
        self.assertNotEqual(eng1, 0)

        # Warm follow-up on session 0 still sticks to engine 0.
        eng_warm, _ = directory.select_engine(
            prompt_token_ids=cold_0 + [77] * 2_048,
            lb_engines=lb,
            inflight_by_idx=[1, 3, 3, 3],
            client_count=1,
            eng_start_index=0,
            now=102.0,
        )
        self.assertEqual(eng_warm, 0)

        # Once engine 0 reports completely idle (0 waiting, 0 running, 0 inflight),
        # its estimated prefill backlog resets to 0 immediately.
        directory.select_engine(
            prompt_token_ids=[300_000 + i for i in range(1024)],
            lb_engines=[(0, 0, 0.1), (0, 2, 0.1), (0, 2, 0.1), (0, 2, 0.1)],
            inflight_by_idx=[0, 2, 2, 2],
            client_count=1,
            eng_start_index=0,
            now=103.0,
        )
        self.assertEqual(directory._engine_prefill_backlog[0], 0.0)

    def test_multi_apiserver_shm_shares_prefix_backlog_and_inflight(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shm_path = os.path.join(tmpdir, "prefix_multi.shm")
            num_clients = 16
            num_engines = 4
            dirs = [
                _DPPrefixDirectory(
                    num_engines=num_engines, chunk_tokens=256, shm_path=shm_path
                )
                for _ in range(num_clients)
            ]
            local_inflight = [[0] * num_engines for _ in range(num_clients)]
            try:
                # 1. APIServer #0 dispatches an 80K cold prompt to engine 0;
                #    APIServer #1 (whose own local_inflight is [0,0,0,0]) must
                #    NOT zero engine 0's backlog and must route its 80K cold
                #    prompt to a different DP engine.
                cold_a = [10_000 + i for i in range(81_920)]
                e0, d0 = dirs[0].select_engine(
                    prompt_token_ids=cold_a,
                    lb_engines=[(0, 0, 0.0)] * num_engines,
                    inflight_by_idx=local_inflight[0],
                    client_count=num_clients,
                    eng_start_index=0,
                    client_index=0,
                    now=10.0,
                )
                self.assertEqual(e0, 0)
                local_inflight[0][e0] += 1
                dirs[0].record(
                    d0,
                    e0,
                    prompt_len=len(cold_a),
                    now=10.0,
                    client_index=0,
                    inflight_by_idx=local_inflight[0],
                )

                cold_b = [200_000 + i for i in range(81_920)]
                e1, d1 = dirs[1].select_engine(
                    prompt_token_ids=cold_b,
                    lb_engines=[(0, 0, 0.0)] * num_engines,
                    inflight_by_idx=local_inflight[1],
                    client_count=num_clients,
                    eng_start_index=0,
                    client_index=1,
                    now=10.005,
                )
                self.assertNotEqual(e1, e0)
                local_inflight[1][e1] += 1
                dirs[1].record(
                    d1,
                    e1,
                    prompt_len=len(cold_b),
                    now=10.005,
                    client_index=1,
                    inflight_by_idx=local_inflight[1],
                )

                # 2. Follow-up turn for session A arrives on APIServer #13 ->
                #    must stick 100% to e0 recorded by APIServer #0!
                e_follow, _ = dirs[13].select_engine(
                    prompt_token_ids=cold_a + [777] * 1024,
                    lb_engines=[(2, 10, 0.3)] * num_engines,
                    inflight_by_idx=local_inflight[13],
                    client_count=num_clients,
                    eng_start_index=2,
                    client_index=13,
                    now=12.0,
                )
                self.assertEqual(e_follow, e0)

                # 3. Binomial variance immunity: suppose coordinator reports
                #    running=20 on all engines and APIServer #0 happens to hold
                #    3 of those 20 decodes on engine 2 (while global_inflight[2]=3<=20).
                #    eff_waiting on engine 2 must stay 0 (no 16*3=48 fake penalty).
                local_inflight[0] = [0, 0, 3, 0]
                local_inflight[1] = [0, 0, 0, 0]
                dirs[1].sync_inflight(1, local_inflight[1])
                dirs[0].reset()
                e_binom, _ = dirs[0].select_engine(
                    prompt_token_ids=[500_000 + i for i in range(1024)],
                    lb_engines=[(0, 25, 0.2), (0, 25, 0.2), (0, 20, 0.2), (0, 25, 0.2)],
                    inflight_by_idx=local_inflight[0],
                    client_count=num_clients,
                    eng_start_index=0,
                    client_index=0,
                    now=20.0,
                )
                self.assertEqual(e_binom, 2)
            finally:
                for d in dirs:
                    d.close()

    def test_patch_multi_client_block_size_16_and_process_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shm_path = os.path.join(tmpdir, "patch_multi.shm")

            class FakeMultiClient:
                def __init__(self, client_index: int, client_count: int = 16):
                    self.vllm_config = SimpleNamespace(
                        cache_config=SimpleNamespace(
                            enable_prefix_caching=True, block_size=16
                        )
                    )
                    self.core_engines = ["e0", "e1", "e2", "e3"]
                    self.lb_engines = [[0, 0, 0.0] for _ in range(4)]
                    self.engine_inflight = Counter()
                    self.client_count = client_count
                    self.client_index = client_index
                    self.eng_start_index = 0
                    self.reqs_in_flight = {}

                def get_core_engine_for_request(self, request):
                    idx = (
                        request.data_parallel_rank
                        if request.data_parallel_rank is not None
                        else 0
                    )
                    eng = self.core_engines[idx]
                    self.reqs_in_flight[request.request_id] = eng
                    self.engine_inflight[eng] += 1
                    return eng

                @staticmethod
                async def process_engine_outputs(self, outputs):
                    for req_id in outputs.finished_requests:
                        eng = self.reqs_in_flight.pop(req_id, None)
                        if eng is not None:
                            self.engine_inflight[eng] -= 1

                async def reset_prefix_cache_async(self):
                    return True

            fake_mod = SimpleNamespace(DPLBAsyncMPClient=FakeMultiClient)
            fake_eng = SimpleNamespace(core_client=fake_mod)
            fake_v1 = SimpleNamespace(engine=fake_eng)
            with patch.dict("os.environ", {"TPU_DP_PREFIX_SHM_PATH": shm_path}):
                with patch.dict(
                    "sys.modules",
                    {
                        "vllm": SimpleNamespace(v1=fake_v1),
                        "vllm.v1": fake_v1,
                        "vllm.v1.engine": fake_eng,
                        "vllm.v1.engine.core_client": fake_mod,
                    },
                ):
                    patch_dp_prefix_routing()

                c0 = FakeMultiClient(client_index=0, client_count=16)
                c1 = FakeMultiClient(client_index=1, client_count=16)
                try:
                    p0 = [9000 + i for i in range(16_384)]
                    r0 = SimpleNamespace(
                        data_parallel_rank=None,
                        resumable=False,
                        request_id="req-0",
                        prompt_token_ids=p0,
                    )
                    eng0 = c0.get_core_engine_for_request(r0)
                    # raw_bs=16 must promote chunk_tokens to 256
                    self.assertEqual(c0._tpu_dp_prefix_dir.chunk_tokens, 256)

                    # Follow-up on c1 routes to same engine eng0
                    r1 = SimpleNamespace(
                        data_parallel_rank=None,
                        resumable=False,
                        request_id="req-1",
                        prompt_token_ids=p0 + [42] * 512,
                    )
                    self.assertEqual(c1.get_core_engine_for_request(r1), eng0)

                    # Staticmethod process_engine_outputs decrements shm inflight cleanly
                    handler = getattr(c0.__class__, "process_engine_outputs")
                    asyncio.run(
                        handler(c0, SimpleNamespace(finished_requests=["req-0"]))
                    )
                    asyncio.run(
                        handler(c1, SimpleNamespace(finished_requests=["req-1"]))
                    )
                    self.assertEqual(
                        c0._tpu_dp_prefix_dir._shm_store.read_global_inflight_unlocked(16),
                        [0, 0, 0, 0],
                    )
                    self.assertTrue(asyncio.run(c1.reset_prefix_cache_async()))
                    self.assertEqual(
                        c0._tpu_dp_prefix_dir.cached_tokens_by_engine(
                            _rolling_chunk_digests(p0, 256)
                        ),
                        [0, 0, 0, 0],
                    )
                finally:
                    if getattr(c0, "_tpu_dp_prefix_dir", None):
                        c0._tpu_dp_prefix_dir.close()
                    if getattr(c1, "_tpu_dp_prefix_dir", None):
                        c1._tpu_dp_prefix_dir.close()


if __name__ == "__main__":
    unittest.main()
