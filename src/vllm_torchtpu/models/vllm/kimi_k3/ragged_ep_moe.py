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
"""Ragged all-to-all expert-parallel MoE dispatch and combine for Kimi K3.1."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec as P

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.ep_mesh import (
    EP_AXIS_NAME,
    build_ep_mesh,
    ep_mesh_index,
    ep_rank_order,
)
from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
from vllm_torchtpu.kernels.fused_moe.v2.layer import _relabel_expert_ids_to_mesh_order
from vllm_torchtpu.kernels.megablox.gmm_v2 import get_packing_factor
from vllm_torchtpu.layers.core.fused_moe_gmm import (
    _counting_sort_positions,
    gmm_wrapper,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

RAGGED_EP_MOE_OP_ATTR = "_ragged_ep_moe_op"
_OPS: dict[tuple[Any, ...], Any] = {}
_RANK_BUFFERS: dict[tuple[torch.device, int], torch.Tensor] = {}


def _mxu_expert_histogram(sort_keys: jax.Array, num_experts: int) -> jax.Array:
    """Count occurrences of ``0..num_experts-1`` in ``sort_keys`` on MXU."""
    m = sort_keys.shape[0]
    blk = min(512, max(1, m))
    pad = (-m) % blk
    if pad:
        sort_keys = jnp.pad(sort_keys, (0, pad), constant_values=num_experts)
    onehot = jax.nn.one_hot(
        sort_keys.reshape(-1, blk), num_experts, dtype=jnp.bfloat16
    )
    ones = jnp.ones((blk,), dtype=jnp.bfloat16)
    counts = jnp.einsum(
        "j,bje->be", ones, onehot, preferred_element_type=jnp.float32
    ).sum(axis=0)
    return counts.astype(jnp.int32)


def _compute_recv_capacity(
    t_local: int,
    topk: int,
    ep_size: int,
    e_local: int,
    max_routes_override: int,
) -> int:
    """Static receive-buffer row capacity specialized for this token bucket."""
    m_send = t_local * topk
    max_possible = max(16, ep_size * t_local * min(topk, e_local))
    if m_send <= 512:
        raw = max(256, m_send * 4)
        aligned = ((raw + 127) // 128) * 128
        return min(max_possible, aligned) if max_possible >= 128 else max_possible
    tile_m = max(128, int(os.environ.get("TPU_GMM_V6E_PREFILL_TILE_M") or 1024))
    raw = max_routes_override if max_routes_override > 0 else (m_send * 5) // 2
    aligned = ((max(tile_m, raw) + tile_m - 1) // tile_m) * tile_m
    return min(max(tile_m, max_possible), aligned)


def _sort_local_routes(
    sort_keys: jax.Array, num_buckets: int
) -> tuple[jax.Array, jax.Array]:
    """Return ``(sorted_pos, inv_perm)`` sorting ``sort_keys`` stably."""
    m = sort_keys.shape[0]
    flat_bits = max(1, (m - 1).bit_length())
    pos = jnp.arange(m, dtype=jnp.int32)
    if num_buckets.bit_length() + flat_bits < 31:
        packed = jax.lax.sort(
            jnp.left_shift(sort_keys, flat_bits) | pos, is_stable=False
        )
        sorted_pos = jnp.bitwise_and(packed, (1 << flat_bits) - 1)
    else:
        sorted_pos = jnp.argsort(sort_keys)
    inv_perm = jnp.zeros_like(pos).at[sorted_pos].set(pos, unique_indices=True)
    return sorted_pos, inv_perm


def make_ragged_ep_moe_local_fn(
    *,
    ep_size: int,
    e_local: int,
    topk: int,
    activation: str,
    rhs_quant_dtype: Any,
    mesh_expert_order: tuple[int, ...] | None,
    max_routes_override: int,
    use_int8: bool,
    per_peer: bool,
    axis_name: str = EP_AXIS_NAME,
    a2a_fn: Callable[..., jax.Array] = jax.lax.ragged_all_to_all,
    expert_compute_fn: Callable[..., jax.Array] | None = None,
) -> Callable[..., jax.Array]:
    """Build the per-shard SPMD body for ``jax.shard_map``."""
    e_total = ep_size * e_local

    def _default_expert_compute(
        recv_x: jax.Array,
        w1_l: jax.Array,
        w2_l: jax.Array,
        w1_scale_l: jax.Array,
        w2_scale_l: jax.Array,
        group_sizes: jax.Array,
    ) -> jax.Array:
        if w1_scale_l.ndim == 3:
            w1_scale_l = jnp.expand_dims(w1_scale_l, 2)
        if w2_scale_l.ndim == 3:
            w2_scale_l = jnp.expand_dims(w2_scale_l, 2)
        hidden_size = recv_x.shape[1]
        packing_factor = get_packing_factor(w1_l.dtype, rhs_quant_dtype)
        padded_hidden_size = w1_l.shape[1] * packing_factor
        if padded_hidden_size > hidden_size:
            recv_x = jnp.pad(recv_x, ((0, 0), (0, padded_hidden_size - hidden_size)))
        group_offset = jnp.array([0], dtype=jnp.int32)
        gmm1_res = gmm_wrapper(
            recv_x,
            w1_l,
            w1_scale_l,
            None,
            group_sizes,
            group_offset,
            zero_initialize=False,
            fuse_act=activation,
            rhs_quant_dtype=rhs_quant_dtype,
        )
        w2_pack = get_packing_factor(w2_l.dtype, rhs_quant_dtype)
        gmm1_res = gmm1_res[:, : w2_l.shape[1] * w2_pack]
        gmm2_res = gmm_wrapper(
            gmm1_res,
            w2_l,
            w2_scale_l,
            None,
            group_sizes,
            group_offset,
            zero_initialize=False,
            preferred_element_type=recv_x.dtype,
            rhs_quant_dtype=rhs_quant_dtype,
        )
        return gmm2_res[:, :hidden_size]

    compute_fn = expert_compute_fn or _default_expert_compute

    def local_fn(
        x_l: jax.Array,
        weights_l: jax.Array,
        ids_l: jax.Array,
        w1_l: jax.Array,
        w2_l: jax.Array,
        w1_scale_l: jax.Array,
        w2_scale_l: jax.Array,
        rank_l: jax.Array,
    ) -> jax.Array:
        me = rank_l[0, 0]
        t_local, hidden = x_l.shape
        out_dtype = x_l.dtype
        m_send = t_local * topk
        r_cap = _compute_recv_capacity(
            t_local, topk, ep_size, e_local, max_routes_override
        )

        mapped_ids = _relabel_expert_ids_to_mesh_order(
            ids_l.astype(jnp.int32),
            g_local=e_local,
            mesh_ep_ranks=mesh_expert_order,
        )
        ids_flat = mapped_ids.reshape(-1)
        w_flat = weights_l.reshape(-1)
        valid = (w_flat != 0) & (ids_flat >= 0) & (ids_flat < e_total)
        sort_keys = jnp.where(valid, ids_flat, e_total)

        sorted_pos, inv_perm = _sort_local_routes(sort_keys, e_total + 1)
        tok_idx = sorted_pos // topk

        send_sizes = _mxu_expert_histogram(sort_keys, e_total)
        counts_all = jax.lax.all_gather(
            send_sizes, axis_name, axis=0, tiled=False
        )  # [ep, e_total]
        counts_sde = counts_all.reshape(ep_size, ep_size, e_local)
        sender_end = jnp.cumsum(counts_all, axis=1)
        sender_start = sender_end - counts_all  # [ep, e_total]

        if not per_peer:
            counts_des = jnp.transpose(counts_sde, (1, 2, 0)).reshape(
                ep_size, e_local * ep_size
            )
            recv_cum_end = jnp.cumsum(counts_des, axis=1)
            recv_cum_start = recv_cum_end - counts_des
            clamped_start = jnp.minimum(recv_cum_start, r_cap).reshape(
                ep_size, e_local, ep_size
            )
            clamped_end = jnp.minimum(recv_cum_end, r_cap).reshape(
                ep_size, e_local, ep_size
            )
            eff_des = clamped_end - clamped_start
            eff_sde = jnp.transpose(eff_des, (2, 0, 1))

            fwd_in_off = sender_start[me]
            fwd_send_sz = eff_sde[me].reshape(ep_size * e_local)
            fwd_out_off = clamped_start[:, :, me].reshape(ep_size * e_local)
            fwd_recv_sz = eff_sde[:, me, :].reshape(ep_size * e_local)

            rev_in_off = jnp.transpose(clamped_start[me], (1, 0)).reshape(
                ep_size * e_local
            )
            rev_send_sz = fwd_recv_sz
            rev_out_off = sender_start.reshape(ep_size, ep_size, e_local)[
                :, me, :
            ].reshape(ep_size * e_local)
            rev_recv_sz = fwd_send_sz
            group_sizes = eff_des[me].sum(axis=1)
        else:
            counts_dse = jnp.transpose(counts_sde, (1, 0, 2)).reshape(
                ep_size, ep_size * e_local
            )
            dse_end = jnp.cumsum(counts_dse, axis=1)
            dse_start = dse_end - counts_dse
            eff_dse = (
                jnp.minimum(dse_end, r_cap) - jnp.minimum(dse_start, r_cap)
            ).reshape(ep_size, ep_size, e_local)
            peer_eff_ds = eff_dse.sum(axis=2)
            peer_eff_sd = jnp.transpose(peer_eff_ds, (1, 0))
            peer_recv_end = jnp.cumsum(peer_eff_ds, axis=1)
            peer_recv_start = peer_recv_end - peer_eff_ds
            peer_sender_start = sender_start.reshape(ep_size, ep_size, e_local)[
                :, :, 0
            ]

            fwd_in_off = peer_sender_start[me]
            fwd_send_sz = peer_eff_sd[me]
            fwd_out_off = peer_recv_start[:, me]
            fwd_recv_sz = peer_eff_ds[me]

            rev_in_off = peer_recv_start[me]
            rev_send_sz = peer_eff_ds[me]
            rev_out_off = peer_sender_start[:, me]
            rev_recv_sz = peer_eff_sd[me]
            group_sizes = eff_dse[me].sum(axis=0)

        if use_int8:
            x_f32 = x_l.astype(jnp.float32)
            amax = jnp.max(jnp.abs(x_f32), axis=-1, keepdims=True)
            row_scale = jnp.maximum(amax * (1.0 / 127.0), 1e-12)
            q_l = jnp.clip(jnp.round(x_f32 / row_scale), -127.0, 127.0).astype(
                jnp.int8
            )
            send_q = q_l[tok_idx]
            send_s = row_scale[tok_idx]
            recv_q = a2a_fn(
                send_q,
                jnp.zeros((r_cap, hidden), dtype=jnp.int8),
                fwd_in_off,
                fwd_send_sz,
                fwd_out_off,
                fwd_recv_sz,
                axis_name=axis_name,
            )
            recv_s = a2a_fn(
                send_s,
                jnp.zeros((r_cap, 1), dtype=jnp.float32),
                fwd_in_off,
                fwd_send_sz,
                fwd_out_off,
                fwd_recv_sz,
                axis_name=axis_name,
            )
            recv_buf = (recv_q.astype(jnp.float32) * recv_s).astype(out_dtype)
        else:
            send_buf = x_l[tok_idx]
            recv_buf = a2a_fn(
                send_buf,
                jnp.zeros((r_cap, hidden), dtype=out_dtype),
                fwd_in_off,
                fwd_send_sz,
                fwd_out_off,
                fwd_recv_sz,
                axis_name=axis_name,
            )

        if per_peer:
            eff_se = eff_dse[me].reshape(ep_size * e_local)
            run_ends = jnp.cumsum(eff_se)
            run_idx = jnp.searchsorted(
                run_ends, jnp.arange(r_cap, dtype=jnp.int32), side="right"
            )
            row_e = jnp.where(
                run_idx < ep_size * e_local,
                jnp.remainder(run_idx, e_local),
                e_local,
            )
            inv_local, counts_local = _counting_sort_positions(row_e, e_local + 1)
            if inv_local is not None and counts_local is not None:
                r_pos = jnp.arange(r_cap, dtype=jnp.int32)
                fwd_local = (
                    jnp.zeros_like(r_pos)
                    .at[inv_local]
                    .set(r_pos, unique_indices=True)
                )
            else:
                fwd_local, inv_local = _sort_local_routes(row_e, e_local + 1)
            gmm_in = recv_buf[fwd_local]
            gmm_sorted = compute_fn(
                gmm_in, w1_l, w2_l, w1_scale_l, w2_scale_l, group_sizes
            )
            gmm_out = gmm_sorted[inv_local]
        else:
            gmm_out = compute_fn(
                recv_buf, w1_l, w2_l, w1_scale_l, w2_scale_l, group_sizes
            )

        back_buf = a2a_fn(
            gmm_out,
            jnp.zeros((m_send, hidden), dtype=out_dtype),
            rev_in_off,
            rev_send_sz,
            rev_out_off,
            rev_recv_sz,
            axis_name=axis_name,
        )
        unpermuted = back_buf[inv_perm].reshape(t_local, topk, hidden)
        combined = (
            unpermuted.astype(jnp.float32)
            * weights_l[:, :, None].astype(jnp.float32)
        ).sum(axis=1)
        return combined.astype(out_dtype)

    return local_fn


def _get_rank_buffer(device: torch.device, mesh_idx: int) -> torch.Tensor:
    key = (device, mesh_idx)
    buf = _RANK_BUFFERS.get(key)
    if buf is None:
        buf = torch.tensor([[mesh_idx]], dtype=torch.int32, device=device)
        _RANK_BUFFERS[key] = buf
    return buf


def prebuild_ragged_ep_moe(
    layer: Any,
    *,
    activation: str,
    rhs_quant_dtype: Any,
) -> Any | None:
    """Build and cache the ragged-all-to-all EP MoE op at weight load time."""
    if not envs.TPU_K3_GENERIC_SP_RAGGED_A2A:
        return None
    parallel_cfg = layer.moe_config.moe_parallel_config
    if not parallel_cfg.use_ep:
        return None
    if int(parallel_cfg.tp_size) != 1:
        logger.warning_once(
            "Ragged EP MoE not engaged: requires moe_tp_size=1, got %d.",
            int(parallel_cfg.tp_size),
        )
        return None
    mesh = build_ep_mesh()
    if mesh is None:
        return None
    ep_size = int(mesh.shape[EP_AXIS_NAME])
    e_local = int(layer.w13_weight.shape[0])
    if int(layer.global_num_experts) != ep_size * e_local:
        logger.warning_once(
            "Ragged EP MoE not engaged: global_num_experts=%d != ep=%d * e_local=%d.",
            int(layer.global_num_experts),
            ep_size,
            e_local,
        )
        return None
    mesh_idx = ep_mesh_index()
    if mesh_idx is None:
        return None
    rank_buf = _get_rank_buffer(layer.w13_weight.device, int(mesh_idx))

    order = ep_rank_order()
    mesh_expert_order = (
        None if order is None or tuple(order) == tuple(range(ep_size)) else tuple(order)
    )
    topk = int(layer.moe_config.experts_per_token)
    max_routes_override = int(envs.VLLM_TPU_MOE_EP_MAX_ROUTES)
    use_int8 = bool(envs.TPU_K3_GENERIC_SP_RAGGED_A2A_INT8)
    per_peer = bool(envs.TPU_K3_GENERIC_SP_RAGGED_A2A_PER_PEER)

    key = (
        tuple(int(d.id) for d in mesh.devices.flat),
        ep_size,
        e_local,
        topk,
        activation,
        rhs_quant_dtype,
        mesh_expert_order,
        max_routes_override,
        use_int8,
        per_peer,
    )
    cached = _OPS.get(key)
    if cached is not None:
        return (cached, rank_buf)

    local_fn = make_ragged_ep_moe_local_fn(
        ep_size=ep_size,
        e_local=e_local,
        topk=topk,
        activation=activation,
        rhs_quant_dtype=rhs_quant_dtype,
        mesh_expert_order=mesh_expert_order,
        max_routes_override=max_routes_override,
        use_int8=use_int8,
        per_peer=per_peer,
    )
    in_specs = (
        P(EP_AXIS_NAME, None),
        P(EP_AXIS_NAME, None),
        P(EP_AXIS_NAME, None),
        P(EP_AXIS_NAME),
        P(EP_AXIS_NAME),
        P(EP_AXIS_NAME),
        P(EP_AXIS_NAME),
        P(EP_AXIS_NAME, None),
    )
    out_spec = P(EP_AXIS_NAME, None)

    def _spmd_moe(
        latent: jax.Array,
        weights: jax.Array,
        ids: jax.Array,
        w1: jax.Array,
        w2: jax.Array,
        w1_scale: jax.Array,
        w2_scale: jax.Array,
        rank: jax.Array,
    ) -> jax.Array:
        return jax.shard_map(
            local_fn,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_spec,
            check_vma=False,
        )(latent, weights, ids, w1, w2, w1_scale, w2_scale, rank)

    op_name = (
        "pallas::kimi_ragged_ep_moe"
        if not _OPS
        else f"pallas::kimi_ragged_ep_moe_{len(_OPS)}"
    )
    op = sharded_jax_op(
        op_name,
        _spmd_moe,
        mesh=mesh,
        input_partition_specs=in_specs,
        output_partition_specs=out_spec,
    )
    op.register_fake(lambda latent, *_args: torch.empty_like(latent))
    _OPS[key] = op
    logger.info_once(
        "Ragged EP MoE armed | ep=%d e_local=%d topk=%d act=%s int8=%s per_peer=%s max_routes=%d",
        ep_size,
        e_local,
        topk,
        activation,
        use_int8,
        per_peer,
        max_routes_override,
    )
    return (op, rank_buf)


def ragged_ep_moe_supported(owner: Any) -> bool:
    return getattr(owner, RAGGED_EP_MOE_OP_ATTR, None) is not None


def run_ragged_ep_moe(
    owner: Any,
    layer: Any,
    latent: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Execute one expert-parallel MoE dispatch + GMM + combine via ragged A2A."""
    op, rank_buf = getattr(owner, RAGGED_EP_MOE_OP_ATTR)
    topk_ids = topk_ids.to(torch.int32)
    topk_weights = topk_weights.to(torch.float32)
    return op(
        latent,
        topk_weights,
        topk_ids,
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale,
        layer.w2_weight_scale,
        rank_buf,
    )
