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
"""Torch bridges for K3's sequence-parallel prefill collectives."""

from functools import lru_cache
from typing import Any

import jax
import torch
from jax.sharding import PartitionSpec as P
from vllm.distributed import get_tp_group

from vllm_torchtpu.distributed.ep_mesh import build_ep_mesh, ep_rank_order, get_ep_group
from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR

from vllm_torchtpu.kernels.kimi_k3.collectives import (  # isort: skip
    gather_moe_inputs,
    gather_project,
    mla_chip_layout,
    mla_preprocess,
    pack_mla_gate,
    project_reduce_scatter,
    ring_collective,
    ring_tables,
)

_OPS: dict[tuple, Any] = {}


def _build_mla_ops(mesh, owners, layer, eps):
    key = (
        "mla_preprocess",
        layer,
        eps,
        tuple(d.id for d in mesh.devices.flat),
        tuple(owners),
    )
    if key in _OPS:
        return _OPS[key]
    pairs, _, _ = mla_chip_layout(list(mesh.devices.flat), owners)

    def pack(weight: jax.Array) -> jax.Array:
        return pack_mla_gate(weight, mesh, pairs)

    pack_op = sharded_jax_op(
        f"pallas::kimi_mla_pack_gate_{layer}",
        pack,
        mesh=mesh,
        input_partition_specs=(P("k3_sp", None),),
        output_partition_specs=P("k3_sp", None),
    )
    pack_op.register_fake(lambda weight: weight.new_empty((7168, 6144)))

    def preprocess(
        x: jax.Array,
        weight: jax.Array,
        packed_gate: jax.Array,
        qnorm: jax.Array,
        kvnorm: jax.Array,
        sequence_parallel: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        return mla_preprocess(
            x,
            weight,
            packed_gate,
            qnorm,
            kvnorm,
            mesh,
            owners,
            sequence_parallel=sequence_parallel,
            eps=eps,
        )

    op = sharded_jax_op(
        f"pallas::kimi_mla_preprocess_{layer}",
        preprocess,
        mesh=mesh,
        input_partition_specs=(P("k3_sp", None),) * 3 + (P(), P(), P()),
        output_partition_specs=(P("k3_sp", None),) * 4,
    )

    def fake(x, weight, packed_gate, qnorm, kvnorm, sequence_parallel):
        rows = x.shape[0] * 32 if sequence_parallel else x.shape[0]
        return tuple(x.new_empty((rows, width)) for width in (1536, 512, 64, 384))

    op.register_fake(fake)
    _OPS[key] = pack_op, op
    return pack_op, op


def _build_moe_gather_op(mesh, tables, collective_id):
    key = (
        "moe_gather",
        collective_id,
        tuple(d.id for d in mesh.devices.flat),
        tuple(tables.ravel()),
    )
    if key in _OPS:
        return _OPS[key]

    def gather(
        x: jax.Array, weights: jax.Array, ids: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return gather_moe_inputs(
            x, weights, ids, mesh, tables, collective_id=collective_id
        )

    op = sharded_jax_op(
        f"pallas::kimi_moe_ag_{collective_id}",
        gather,
        mesh=mesh,
        input_partition_specs=(P("k3_sp", None),) * 3,
        output_partition_specs=(P(),) * 3,
    )

    def fake(x, weights, ids):
        return tuple(
            a.new_empty((a.shape[0] * 32, a.shape[1])) for a in (x, weights, ids)
        )

    op.register_fake(fake)
    _OPS[key] = op
    return op


def _build_op(mesh, tables, gather, collective_id):
    key = (
        gather,
        collective_id,
        tuple(d.id for d in mesh.devices.flat),
        tuple(tables.ravel()),
    )
    if key in _OPS:
        return _OPS[key]

    def collective(x: jax.Array) -> jax.Array:
        # The bridge assigns a partitioned global shape to this rank's input.
        # RS represents each rank's full-token partial as one leading slice.
        if not gather:
            x = x.reshape(32, x.shape[0] // 32, x.shape[1])
        return ring_collective(
            x, mesh, tables, gather=gather, collective_id=collective_id
        )

    name = f"pallas::kimi_ring_{'ag' if gather else 'rs'}_{collective_id}"
    op = sharded_jax_op(
        name,
        collective,
        mesh=mesh,
        input_partition_specs=(P("k3_sp", None),),
        output_partition_specs=P() if gather else P("k3_sp", None),
    )

    def fake(x):
        rows = x.shape[0] * 32 if gather else x.shape[0] // 32
        return torch.empty((rows, x.shape[1]), dtype=x.dtype, device=x.device)

    op.register_fake(fake)
    _OPS[key] = op
    return op


def _build_project_op(mesh, tables, collective_id):
    key = (
        "project",
        collective_id,
        tuple(d.id for d in mesh.devices.flat),
        tuple(tables.ravel()),
    )
    if key in _OPS:
        return _OPS[key]

    def project(x: jax.Array, w: jax.Array) -> jax.Array:
        return project_reduce_scatter(
            x.reshape(32, x.shape[0] // 32, 384),
            w.reshape(32, 384, 7168),
            mesh,
            tables,
            collective_id=collective_id,
        )

    op = sharded_jax_op(
        f"pallas::kimi_project_rs_{collective_id}",
        project,
        mesh=mesh,
        input_partition_specs=(P("k3_sp", None), P("k3_sp", None)),
        output_partition_specs=P("k3_sp", None),
    )

    def fake(x, w):
        return torch.empty(
            (x.shape[0] // 32, w.shape[1]), dtype=x.dtype, device=x.device
        )

    op.register_fake(fake)
    _OPS[key] = op
    return op


def _build_gather_project_op(mesh, tables, collective_id):
    key = (
        "gather_project",
        collective_id,
        tuple(d.id for d in mesh.devices.flat),
        tuple(tables.ravel()),
    )
    if key in _OPS:
        return _OPS[key]

    def project(x: jax.Array, w: jax.Array) -> jax.Array:
        result = gather_project(
            x, w.reshape(32, 7168, 1792), mesh, tables, collective_id=collective_id
        )
        return result.reshape(-1, 1792)

    op = sharded_jax_op(
        f"pallas::kimi_gather_project_{collective_id}",
        project,
        mesh=mesh,
        input_partition_specs=(P("k3_sp", None), P("k3_sp", None)),
        output_partition_specs=P("k3_sp", None),
    )

    def fake(x, w):
        return torch.empty(
            (x.shape[0] * 32, w.shape[1]), dtype=x.dtype, device=x.device
        )

    op.register_fake(fake)
    _OPS[key] = op
    return op


@lru_cache(maxsize=1)
def _mesh_and_tables():
    ep = get_ep_group()
    base = get_tp_group()
    if ep is None or list(ep.ranks) != list(base.ranks):
        raise ValueError("K3 rings require identical TP32 and EP32 rank order")
    mesh = build_ep_mesh(axis_name="k3_sp")
    assert mesh is not None
    owners = ep_rank_order()
    return mesh, ring_tables(list(mesh.devices.flat), owners)


@lru_cache(maxsize=4)
def _build_token_op(world_size, gather):
    mesh, tables = _mesh_and_tables()
    owners = tuple(int(i) for i in tables[0, 2])
    if world_size == 2:
        groups, _, _ = mla_chip_layout(list(mesh.devices.flat), owners)
    elif world_size == 32:
        groups = (tuple(sorted(range(32), key=lambda i: owners[i])),)
    else:
        raise ValueError("K3 SP token collectives require TP2 or TP32")

    def collective(x: jax.Array) -> jax.Array:
        def local(part):
            if gather:
                return jax.lax.all_gather(
                    part, "k3_sp", axis=0, tiled=True, axis_index_groups=groups
                )
            return jax.lax.psum_scatter(
                part, "k3_sp", scatter_dimension=0, tiled=True, axis_index_groups=groups
            )

        return jax.shard_map(
            local,
            mesh=mesh,
            in_specs=P("k3_sp", None),
            out_specs=P("k3_sp", None),
            check_vma=False,
        )(x)

    op = sharded_jax_op(
        f"pallas::kimi_token_{world_size}_{'ag' if gather else 'rs'}",
        collective,
        mesh=mesh,
        input_partition_specs=(P("k3_sp", None),),
        output_partition_specs=P("k3_sp", None),
    )

    def fake(x):
        rows = x.shape[0] * world_size if gather else x.shape[0] // world_size
        return x.new_empty((rows, x.shape[1]))

    op.register_fake(fake)
    return op


class TokenShardCollectives:
    """Gather/scatter token shards over TP2 or TP32 without concrete allocations."""

    def __init__(self, base):
        self.world_size = base.world_size
        self.rank_in_group = base.rank_in_group
        self.ag = _build_token_op(self.world_size, True)
        self.rs = _build_token_op(self.world_size, False)

    def all_gather(self, x, dim):
        assert dim == 0
        return self.ag(x)

    def reduce_scatter(self, x, dim):
        assert dim == 0
        return self.rs(x)


class FusedPrefillCollectives:
    """Fused prefill projections and MoE collectives, with standard fallbacks."""

    def __init__(self, prefix):
        self.base = get_tp_group()
        mesh, tables = _mesh_and_tables()
        # Fixed layer-derived IDs keep independent ring barriers distinct.
        components = prefix.split(".")
        layer = int(components[components.index("layers") + 1])
        self.layer = layer
        self.ag = _build_op(mesh, tables, True, 20000 + 2 * layer)
        self.rs = _build_op(mesh, tables, False, 20001 + 2 * layer)
        self.project_rs = _build_project_op(mesh, tables, 22000 + layer)
        self.gather_project = _build_gather_project_op(mesh, tables, 23000 + layer)
        self._moe_gather_op = _build_moe_gather_op(mesh, tables, 24000 + layer)

    def gather_moe_inputs(self, latent, weights, ids, *, num_experts):
        """Gather token activations and routes, packing compatible BF16 inputs."""
        weights, ids = weights.to(latent.dtype), ids.to(torch.int32)
        if num_experts <= 32768 and weights.dtype == torch.bfloat16:
            return self._moe_gather_op(latent, weights, ids)
        return tuple(self.all_gather(x, dim=0) for x in (latent, weights, ids))

    def mla_ops(self, eps):
        mesh, _ = _mesh_and_tables()
        return _build_mla_ops(mesh, ep_rank_order(), self.layer, eps)

    def all_gather(self, x, dim):
        if (
            dim == 0
            and x.ndim == 2
            and x.shape[0] > 0
            and x.shape[1] in (2176, 3584, 7168)
            and x.dtype == torch.bfloat16
        ):
            return self.ag(x)
        return self.base.all_gather(x, dim=dim)

    def reduce_scatter(self, x, dim):
        if (
            dim == 0
            and x.ndim == 2
            and x.shape[0] > 0
            and x.shape[0] % 32 == 0
            and x.shape[1] in (2176, 3584, 7168)
            and x.dtype == torch.bfloat16
        ):
            return self.rs(x)
        return self.base.reduce_scatter(x, dim=dim)

    def project_reduce_scatter(self, x, projection):
        # Preserve the linear method for other weight formats.
        weight = getattr(projection, "weight", None)
        if (
            x.shape[0] > 0
            and x.shape[0] % 32 == 0
            and x.shape[1] == 384
            and x.dtype == torch.bfloat16
            and weight is not None
            and weight.dtype == torch.bfloat16
            and weight.shape == (384, 7168)
            and getattr(projection, WEIGHT_FLIPPED_ATTR, False)
            and getattr(projection, "bias", None) is None
        ):
            return self.project_rs(x, weight)
        return self.reduce_scatter(projection(x)[0], dim=0)

    def all_reduce(self, x):
        return self.base.all_reduce(x)


class _ProcessGroupCollectiveAdapter:
    """GroupCoordinator-compatible dim-0 all_gather / reduce_scatter over a ProcessGroup."""

    def __init__(self, pg: Any, world_size: int) -> None:
        self.device_group = pg
        self.world_size = int(world_size)

    def all_gather(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        import torch.distributed._functional_collectives as funcol

        if self.world_size <= 1:
            return x
        if dim < 0:
            dim += x.dim()
        assert dim == 0, f"_ProcessGroupCollectiveAdapter only supports dim=0, got {dim}"
        out = funcol.all_gather_single(x.contiguous(), 0, self.device_group)
        if isinstance(out, funcol.AsyncCollectiveTensor):
            out = out.wait()
        return out

    def reduce_scatter(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        import torch.distributed._functional_collectives as funcol

        if self.world_size <= 1:
            return x
        if dim < 0:
            dim += x.dim()
        assert dim == 0, f"_ProcessGroupCollectiveAdapter only supports dim=0, got {dim}"
        out = funcol.reduce_scatter_single(x.contiguous(), "sum", 0, self.device_group)
        if isinstance(out, funcol.AsyncCollectiveTensor):
            out = out.wait()
        return out


class GenericSPCollectives:
    """Process-group SP collectives for multi-host / DP>=1 Kimi K3 deployments."""

    def __init__(self, tp_group=None) -> None:
        from vllm_torchtpu import envs
        from vllm_torchtpu.distributed.ep_mesh import ep_sorted_gather_group

        self.base = tp_group if tp_group is not None else get_tp_group()
        self.world_size = int(self.base.world_size)
        self.rank_in_group = int(self.base.rank_in_group)
        self._sorted_ep_group = None
        if envs.TPU_K3_GENERIC_SP_SORTED_EP_GROUP and get_ep_group() is not None:
            sorted_ep = ep_sorted_gather_group()
            if sorted_ep is not None:
                pg, positions = sorted_ep
                self._sorted_ep_group = _ProcessGroupCollectiveAdapter(
                    pg, len(positions)
                )

    def all_gather(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        if self.world_size <= 1:
            return x
        return self.base.all_gather(x, dim=dim)

    def reduce_scatter(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        if self.world_size <= 1:
            return x
        return self.base.reduce_scatter(x, dim=dim)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.world_size <= 1:
            return x
        return self.base.all_reduce(x)

    def _moe_gather_group(self, *, sequence_parallel: bool):
        if sequence_parallel:
            if self._sorted_ep_group is not None:
                return self._sorted_ep_group
            ep = get_ep_group()
            return ep if ep is not None else self.base
        from vllm.distributed.parallel_state import get_dp_group

        try:
            return get_dp_group()
        except (AssertionError, AttributeError, RuntimeError):
            return None

    def gather_moe_inputs(
        self,
        latent: torch.Tensor,
        weights: torch.Tensor,
        ids: torch.Tensor,
        *,
        num_experts: int,
        sequence_parallel: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del num_experts
        from vllm_torchtpu import envs
        from vllm_torchtpu.layers.adapter.pipelined_fused_moe import (
            _quantized_all_gather_with_routing,
        )

        ids = ids.to(torch.int32)
        group = self._moe_gather_group(sequence_parallel=sequence_parallel)
        if group is None or int(group.world_size) <= 1:
            return latent, weights, ids
        if envs.TPU_MOE_FP8_ALL_GATHER:
            ag_latent, ag_weights, ag_ids = _quantized_all_gather_with_routing(
                group, latent, weights, ids
            )
            assert ag_weights is not None and ag_ids is not None
            return ag_latent, ag_weights, ag_ids
        return (
            group.all_gather(latent, dim=0),
            group.all_gather(weights, dim=0),
            group.all_gather(ids, dim=0),
        )

    def reduce_moe_output(
        self, output: torch.Tensor, *, sequence_parallel: bool = True
    ) -> torch.Tensor:
        if sequence_parallel:
            group = self._moe_gather_group(sequence_parallel=True)
            if group is None or int(group.world_size) <= 1:
                return output
            return group.reduce_scatter(output, dim=0)
        dp_group = self._moe_gather_group(sequence_parallel=False)
        if dp_group is not None and int(dp_group.world_size) > 1:
            output = dp_group.reduce_scatter(output, dim=0)
        return self.all_reduce(output)
