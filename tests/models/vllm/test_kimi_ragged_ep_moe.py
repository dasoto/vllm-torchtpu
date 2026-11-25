# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Unit tests for ragged all-to-all expert-parallel Kimi K3.1 MoE."""

from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.vllm.kimi_k3.ragged_ep_moe import (
    _compute_recv_capacity,
    make_ragged_ep_moe_local_fn,
)


def _cpu_ragged_a2a(
    operand,
    output,
    input_offsets,
    send_sizes,
    output_offsets,
    recv_sizes,
    *,
    axis_name,
):
    ep = jax.lax.psum(jnp.int32(1), axis_name)
    n = input_offsets.shape[0]
    spd = n // int(input_offsets.shape[0] // jax.lax.axis_size(axis_name))
    del ep
    spd = n // jax.lax.axis_size(axis_name)
    all_op = jax.lax.all_gather(operand, axis_name, axis=0, tiled=False)
    out_off_recv = jax.lax.all_to_all(
        output_offsets, axis_name, split_axis=0, concat_axis=0, tiled=True
    )
    in_off_from_src = jax.lax.all_to_all(
        input_offsets, axis_name, split_axis=0, concat_axis=0, tiled=True
    )
    expected_send = jax.lax.all_to_all(
        recv_sizes, axis_name, split_axis=0, concat_axis=0, tiled=True
    )

    def body_i(i, res):
        s = i // spd
        src_data = all_op[s]
        ioff = in_off_from_src[i]
        ooff = out_off_recv[i]
        rsz = recv_sizes[i]

        def body_r(r, acc):
            return acc.at[ooff + r].set(src_data[ioff + r])

        return jax.lax.fori_loop(0, rsz, body_r, res)

    res = jax.lax.fori_loop(0, n, body_i, output)
    # Fail with NaN if send_sizes != all_to_all(recv_sizes)
    bad = jnp.any(send_sizes != expected_send)
    return jnp.where(bad, jnp.full_like(res, jnp.nan), res)


def _linear_expert_compute(recv_x, w1_l, _w2_l, _s1, _s2, group_sizes):
    r_cap = recv_x.shape[0]
    exp_ends = jnp.cumsum(group_sizes)
    exp_starts = exp_ends - group_sizes
    rows = jnp.arange(r_cap, dtype=jnp.int32)[:, None]
    mask = (rows >= exp_starts[None, :]) & (rows < exp_ends[None, :])
    out = jnp.einsum(
        "re,rh,eho->ro",
        mask.astype(jnp.float32),
        recv_x.astype(jnp.float32),
        w1_l.astype(jnp.float32),
    )
    return out.astype(recv_x.dtype)


def _ref_moe(x, weights, ids, w_global):
    out = jnp.zeros_like(x, dtype=jnp.float32)
    for t in range(x.shape[0]):
        for k in range(weights.shape[1]):
            w = weights[t, k]
            e = int(ids[t, k])
            if float(w) != 0.0:
                out = out.at[t].add(
                    w * (x[t].astype(jnp.float32) @ w_global[e].astype(jnp.float32))
                )
    return out.astype(x.dtype)


@pytest.mark.parametrize("per_peer", [False, True])
@pytest.mark.parametrize("use_int8", [False, True])
@pytest.mark.parametrize("permute_mesh", [False, True])
def test_ragged_ep_moe_numerical_parity(per_peer: bool, use_int8: bool, permute_mesh: bool):
    devs = jax.devices("cpu")
    if len(devs) < 4:
        pytest.skip("Requires >= 4 host CPU devices")
    ep = 4
    mesh = Mesh(np.array(devs[:ep]), ("d",))
    e_local = 3
    e_total = ep * e_local
    topk = 4
    t_local = 12
    hidden = 32
    order = (2, 0, 3, 1) if permute_mesh else None

    key = jax.random.PRNGKey(17)
    k1, k2, k3 = jax.random.split(key, 3)
    x = jax.random.normal(k1, (ep * t_local, hidden), dtype=jnp.bfloat16)
    logits = jax.random.normal(k2, (ep * t_local, e_total), dtype=jnp.float32)
    top_v, top_i = jax.lax.top_k(jax.nn.sigmoid(logits), topk)
    top_w = top_v / top_v.sum(axis=-1, keepdims=True)
    pad_mask = (jnp.arange(ep * t_local) % t_local) >= (t_local - 2)
    top_w = jnp.where(pad_mask[:, None], 0.0, top_w)

    w_ep_ordered = (
        jax.random.normal(k3, (ep, e_local, hidden, hidden), dtype=jnp.bfloat16) * 0.1
    )
    w_global = w_ep_ordered.reshape(e_total, hidden, hidden)
    if order is not None:
        w_mesh_ordered = w_ep_ordered[jnp.array(order)].reshape(e_total, hidden, hidden)
    else:
        w_mesh_ordered = w_global

    local_fn = make_ragged_ep_moe_local_fn(
        ep_size=ep,
        e_local=e_local,
        topk=topk,
        activation="silu",
        rhs_quant_dtype=None,
        mesh_expert_order=order,
        max_routes_override=256,
        use_int8=use_int8,
        per_peer=per_peer,
        axis_name="d",
        a2a_fn=_cpu_ragged_a2a,
        expert_compute_fn=_linear_expert_compute,
    )
    in_specs = (
        P("d", None),
        P("d", None),
        P("d", None),
        P("d"),
        P("d"),
        P("d"),
        P("d"),
        P("d", None),
    )
    sh2 = NamedSharding(mesh, P("d", None))
    sh_w = NamedSharding(mesh, P("d"))
    ranks = jnp.arange(ep, dtype=jnp.int32).reshape(ep, 1)
    dummy_s = jnp.ones((e_total, 1, 1, hidden), dtype=jnp.float32)

    fn = jax.jit(
        lambda x_, w_, i_, w1_, w2_, s1_, s2_, rk_: jax.shard_map(
            local_fn, mesh=mesh, in_specs=in_specs, out_specs=P("d", None), check_vma=False
        )(x_, w_, i_, w1_, w2_, s1_, s2_, rk_),
        in_shardings=(sh2, sh2, sh2, sh_w, sh_w, sh_w, sh_w, sh2),
        out_shardings=sh2,
    )
    got = fn(
        jax.device_put(x, sh2),
        jax.device_put(top_w, sh2),
        jax.device_put(top_i, sh2),
        jax.device_put(w_mesh_ordered, sh_w),
        jax.device_put(w_mesh_ordered, sh_w),
        jax.device_put(dummy_s, sh_w),
        jax.device_put(dummy_s, sh_w),
        jax.device_put(ranks, sh2),
    )
    ref = _ref_moe(x, top_w, top_i, w_global)
    assert bool(jnp.all(jnp.isfinite(got)))
    tol = 0.03 if use_int8 else 0.01
    max_diff = float(jnp.max(jnp.abs(got.astype(jnp.float32) - ref.astype(jnp.float32))))
    assert max_diff <= tol, f"max_diff={max_diff} > {tol}"
    assert float(jnp.max(jnp.abs(got[pad_mask].astype(jnp.float32)))) == 0.0


def test_ragged_ep_moe_tpu_export_no_partition_id():
    devs = jax.devices("cpu")
    if len(devs) < 4:
        pytest.skip("Requires >= 4 host CPU devices")
    ep = 4
    mesh = Mesh(np.array(devs[:ep]), ("d",))
    e_local = 3
    e_total = ep * e_local
    topk = 4
    t_local = 64
    hidden = 128

    local_fn = make_ragged_ep_moe_local_fn(
        ep_size=ep,
        e_local=e_local,
        topk=topk,
        activation="silu",
        rhs_quant_dtype=None,
        mesh_expert_order=None,
        max_routes_override=1024,
        use_int8=False,
        per_peer=False,
        axis_name="d",
        expert_compute_fn=_linear_expert_compute,
    )
    in_specs = (
        P("d", None),
        P("d", None),
        P("d", None),
        P("d"),
        P("d"),
        P("d"),
        P("d"),
        P("d", None),
    )
    sh2 = NamedSharding(mesh, P("d", None))
    sh_w = NamedSharding(mesh, P("d"))
    fn = jax.jit(
        lambda x_, w_, i_, w1_, w2_, s1_, s2_, rk_: jax.shard_map(
            local_fn, mesh=mesh, in_specs=in_specs, out_specs=P("d", None), check_vma=False
        )(x_, w_, i_, w1_, w2_, s1_, s2_, rk_),
        in_shardings=(sh2, sh2, sh2, sh_w, sh_w, sh_w, sh_w, sh2),
        out_shardings=sh2,
    )
    exported = jax.export.export(
        fn,
        platforms=["tpu"],
        disabled_checks=[jax.export.DisabledSafetyCheck.custom_call("ALL")],
    )(
        jax.ShapeDtypeStruct((ep * t_local, hidden), jnp.bfloat16, sharding=sh2),
        jax.ShapeDtypeStruct((ep * t_local, topk), jnp.float32, sharding=sh2),
        jax.ShapeDtypeStruct((ep * t_local, topk), jnp.int32, sharding=sh2),
        jax.ShapeDtypeStruct((e_total, hidden, hidden), jnp.bfloat16, sharding=sh_w),
        jax.ShapeDtypeStruct((e_total, hidden, hidden), jnp.bfloat16, sharding=sh_w),
        jax.ShapeDtypeStruct((e_total, 1, 1, hidden), jnp.float32, sharding=sh_w),
        jax.ShapeDtypeStruct((e_total, 1, 1, hidden), jnp.float32, sharding=sh_w),
        jax.ShapeDtypeStruct((ep, 1), jnp.int32, sharding=sh2),
    )
    mlir = exported.mlir_module()
    assert "stablehlo.partition_id" not in mlir
    assert "@ragged_all_to_all" in mlir


def test_compute_recv_capacity_decode_vs_prefill():
    assert _compute_recv_capacity(6, 16, 128, 7, 20480) == 384
    assert _compute_recv_capacity(3, 16, 128, 7, 20480) == 256
    assert _compute_recv_capacity(512, 16, 128, 7, 20480) == 20480
    assert _compute_recv_capacity(256, 16, 128, 7, 0) == 10240
