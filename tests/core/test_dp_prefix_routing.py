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

import unittest
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
        with patch.dict("sys.modules", {"vllm.v1.engine.core_client": fake_mod}):
            import vllm.v1.engine as eng_pkg

            with patch.object(eng_pkg, "core_client", fake_mod, create=True):
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


if __name__ == "__main__":
    unittest.main()
