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
"""Kimi MoE routed through vLLM's TPU Pallas fused-MoE backend."""

from __future__ import annotations

import zlib
from contextlib import ExitStack
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import torch
from torch import nn
from vllm.config import get_current_vllm_config_or_none
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    model_parallel_is_initialized,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.collectives import all_to_all
from vllm_torchtpu.distributed.intra_chip import get_intra_chip_group, get_tp_subgroup
from vllm_torchtpu.layers.adapter import moe_routing
from vllm_torchtpu.layers.adapter.latent_proj_intra_chip import (
    IntraChipColumnParallelLinear,
    make_latent_projection,
)
from vllm_torchtpu.layers.adapter.moe_hierarchical import (
    hierarchical_moe_parallel_config,
    hierarchical_split_or_none,
)
from vllm_torchtpu.layers.adapter.quantization.unquantized import VllmUnquantizedConfig

from .collective_ops import FusedPrefillCollectives, GenericSPCollectives
from .layers import KimiMLP

if TYPE_CHECKING:
    from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4 import (  # noqa: E501
        VllmCompressedTensorsW4ANMxfp4MoEMethod,
    )


class KimiRoutedOutputTransform(nn.Module):
    def __init__(
        self,
        norm: RMSNorm | None,
        # Replicated or intra-chip sharded; both return ``(output, bias)``.
        up_proj: nn.Module,
    ) -> None:
        super().__init__()
        self.norm = norm
        self.up_proj = up_proj

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        hidden_states, _ = self.up_proj(hidden_states)
        return hidden_states


def _subgroup_feature_to_token_a2a(
    x: torch.Tensor, group, part: int
) -> torch.Tensor:
    """Redistribute ``[S_local, W * part]`` across ``group`` into ``[W * S_local, part]``."""
    ws = int(group.world_size)
    if ws <= 1:
        return x
    if getattr(group, "device_group", None) is not None:
        return all_to_all(
            x.unflatten(-1, (ws, part)).movedim(0, 1).reshape(-1, part),
            group,
        )
    rank = int(group.rank_in_group)
    return group.all_gather(x, dim=0).narrow(-1, rank * part, part).contiguous()


class KimiMoE(nn.Module):
    """Standard or latent Kimi MoE backed by the fused TPU GMM custom op."""

    def __init__(
        self,
        config: KimiLinearConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        assert config.num_experts is not None
        assert config.num_experts_per_token is not None
        assert config.moe_intermediate_size is not None

        self.prefix = prefix
        hidden_size = config.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = 0
        vllm_config = get_current_vllm_config_or_none()
        self.use_ep = (
            vllm_config is not None
            and vllm_config.parallel_config.enable_expert_parallel
        )
        routed_expert_hidden_size = config.routed_expert_hidden_size
        expert_hidden_size = (
            routed_expert_hidden_size
            if routed_expert_hidden_size is not None
            else hidden_size
        )

        gate_quant_config = VllmUnquantizedConfig() if quant_config is not None else None
        self.gate = GateLinear(
            input_size=hidden_size,
            output_size=config.num_experts,
            bias=False,
            out_dtype=torch.float32,
            quant_config=gate_quant_config,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.zeros(config.num_experts)
        )

        # Replicated by default; sharded across a chip's cores under
        # TPU_LATENT_PROJ_INTRA_CHIP_TP. Pass quant_config so TPU's dense
        # loader transposes to canonical (k, n) layout and honors
        # QUANTIZE_ON_LOAD_PREFIXES. When TP > 1 (non-SP), shard
        # routed_expert_up_proj row-parallel across TP with reduce_results=False
        # and fuse its reduction with shared_experts in one TP all-reduce.
        self.generic_sp = (
            not envs.TPU_K3_SP_PREFILL
            and envs.TPU_K3_GENERIC_SP
            and self.tp_size > 1
            and self.use_ep
        )
        self._fuse_tp_up_proj = (
            not envs.TPU_K3_SP_PREFILL
            and not self.generic_sp
            and routed_expert_hidden_size is not None
            and self.tp_size > 1
            and expert_hidden_size % self.tp_size == 0
        )
        self._tp_rank_buf: torch.Tensor | None
        self.routed_expert_down_proj = (
            make_latent_projection(
                hidden_size,
                expert_hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.routed_expert_down_proj",
            )
            if routed_expert_hidden_size is not None
            else None
        )
        self.routed_expert_norm = (
            RMSNorm(expert_hidden_size, config.rms_norm_eps)
            if self.routed_expert_down_proj is not None and config.latent_moe_use_norm
            else None
        )

        self.prefill_group = None
        self.shared_group = None
        self.sp_moe_chunks = 1
        if envs.TPU_K3_SP_PREFILL:
            self.prefill_group = FusedPrefillCollectives(prefix)
            self.shared_group = get_intra_chip_group()
            if self.shared_group is None or self.shared_group.world_size != 2:
                raise ValueError("K3 SP prefill needs a physical two-core chip group")
        elif self.generic_sp:
            from vllm.distributed import get_tp_group

            self.prefill_group = GenericSPCollectives()
            self.sp_moe_chunks = max(1, envs.TPU_K3_GENERIC_SP_MOE_CHUNKS)
            shared_interm = config.moe_intermediate_size * config.num_shared_experts
            sub_sz = envs.TPU_K3_GENERIC_SP_SHARED_SUBGROUP
            subgroup = (
                get_tp_subgroup(sub_sz)
                if (
                    sub_sz >= 2
                    and sub_sz < self.tp_size
                    and shared_interm > 0
                    and shared_interm % sub_sz == 0
                )
                else None
            )
            self.shared_group = subgroup if subgroup is not None else get_tp_group()

        self._fuse_generic_sp_up_proj = (
            self.generic_sp
            and routed_expert_hidden_size is not None
            and self.shared_group is not None
            and int(self.shared_group.world_size) > 1
            and expert_hidden_size % int(self.shared_group.world_size) == 0
        )

        if self.routed_expert_down_proj is None:
            self.routed_expert_up_proj = None
        elif self._fuse_tp_up_proj:
            self.routed_expert_up_proj = RowParallelLinear(
                expert_hidden_size,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.routed_expert_up_proj",
            )
            up_tp_rank = getattr(self.routed_expert_up_proj, "tp_rank", None)
            self.tp_rank = (
                int(up_tp_rank)
                if up_tp_rank is not None
                else (
                    get_tensor_model_parallel_rank()
                    if model_parallel_is_initialized()
                    else 0
                )
            )
            self.register_buffer(
                "_tp_rank_buf",
                torch.tensor([self.tp_rank], dtype=torch.long),
                persistent=False,
            )
        elif self._fuse_generic_sp_up_proj:
            assert self.shared_group is not None
            with ExitStack() as stack:
                module = "vllm.model_executor.layers.linear"
                stack.enter_context(
                    patch(
                        f"{module}.get_tensor_model_parallel_rank",
                        return_value=self.shared_group.rank_in_group,
                    )
                )
                stack.enter_context(
                    patch(
                        f"{module}.get_tensor_model_parallel_world_size",
                        return_value=self.shared_group.world_size,
                    )
                )
                self.routed_expert_up_proj = RowParallelLinear(
                    expert_hidden_size,
                    hidden_size,
                    bias=False,
                    quant_config=quant_config,
                    reduce_results=False,
                    prefix=f"{prefix}.routed_expert_up_proj",
                )
        else:
            self.routed_expert_up_proj = make_latent_projection(
                expert_hidden_size,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.routed_expert_up_proj",
            )
        if not self._fuse_tp_up_proj:
            self.register_buffer("_tp_rank_buf", None, persistent=False)
        routed_output_transform = (
            KimiRoutedOutputTransform(
                self.routed_expert_norm, self.routed_expert_up_proj
            )
            if (
                self.routed_expert_up_proj is not None
                and not self._fuse_tp_up_proj
                and not self._fuse_generic_sp_up_proj
            )
            else None
        )

        self.shared_experts = (
            KimiMLP(
                hidden_size,
                config.moe_intermediate_size * config.num_shared_experts,
                config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_experts",
                reduce_results=False,
                tp_group=self.shared_group,
                situ_beta=config.activation_situ_beta,
                situ_linear_beta=config.activation_situ_linear_beta,
            )
            if config.num_shared_experts > 0
            else None
        )

        padded_intermediate_size = config.moe_intermediate_size
        min_per_partition = getattr(config, "min_moe_intermediate_per_partition", 128)
        if self.use_ep:
            split = hierarchical_split_or_none()
            moe_tp_size = split[2] if split is not None else 1
        else:
            moe_tp_size = self.tp_size
        if (
            moe_tp_size > 1
            and padded_intermediate_size // moe_tp_size < min_per_partition
        ):
            padded_intermediate_size = min_per_partition * moe_tp_size

        situ_beta = config.activation_situ_beta if config.hidden_act == "situ" else None
        situ_linear_beta = (
            config.activation_situ_linear_beta if config.hidden_act == "situ" else None
        )
        # Under TPU_MOE_HIERARCHICAL_EP this builds the routed experts
        # with expert parallelism between chips and tensor parallelism
        # within one chip, instead of vLLM's flat all-experts-per-device EP.
        with hierarchical_moe_parallel_config():
            self.experts = FusedMoEFactory(
                shared_experts=(
                    None
                    if (self._fuse_tp_up_proj or self._fuse_generic_sp_up_proj)
                    else self.shared_experts
                ),
                num_experts=config.num_experts,
                top_k=config.num_experts_per_token,
                hidden_size=expert_hidden_size,
                intermediate_size=padded_intermediate_size,
                activation=config.hidden_act,
                activation_situ_beta=situ_beta,
                activation_situ_linear_beta=situ_linear_beta,
                renormalize=config.moe_renormalize,
                quant_config=quant_config,
                use_grouped_topk=config.use_grouped_topk,
                num_expert_group=config.num_expert_group,
                topk_group=config.topk_group,
                prefix=f"{prefix}.experts",
                scoring_func=config.moe_router_activation_func,
                e_score_correction_bias=cast(
                    torch.Tensor, self.gate.e_score_correction_bias
                ),
                routed_scaling_factor=config.routed_scaling_factor,
                routed_input_transform=self.routed_expert_down_proj,
                routed_output_transform=routed_output_transform,
            )
        if self.shared_group is not None and not hasattr(
            self.experts.routed_experts.quant_method, "apply_with_routing"
        ):
            raise ValueError("K3 SP prefill requires the native MXFP4 MoE backend")
        if padded_intermediate_size != config.moe_intermediate_size:
            routed_experts = self.experts.routed_experts
            w13_weight = getattr(routed_experts, "w13_weight", None)
            if w13_weight is None:
                w13_weight = cast(torch.Tensor, routed_experts.w13_weight_packed)
            w2_weight = getattr(routed_experts, "w2_weight", None)
            if w2_weight is None:
                w2_weight = cast(torch.Tensor, routed_experts.w2_weight_packed)
            w13_weight.data.zero_()
            w2_weight.data.zero_()
            self.experts.moe_config.intermediate_size_per_partition_unpadded = (
                config.moe_intermediate_size // moe_tp_size
            )

    def initialize_dummy_router(self) -> None:
        """Use fan-in random weights and fixed, distinct random layer biases.

        The generic dummy loader's +/-0.001 weights make the correction bias
        dominate sigmoid routing scores. Match the prototype's unit logit
        variance for normalized inputs without imposing any route counts.
        """
        config = get_current_vllm_config_or_none()
        if config is None or str(config.load_config.load_format).lower() != "dummy":
            return
        generator = torch.Generator(device="cpu").manual_seed(
            1234 + zlib.crc32(self.prefix.encode())
        )
        bias_param = cast(torch.Tensor, self.gate.e_score_correction_bias)
        bias = (
            torch.randn(
                bias_param.shape,
                generator=generator,
                dtype=torch.float32,
            )
            * 0.02
        )
        weight_param = cast(torch.Tensor, self.gate.weight)
        weight = torch.randn(
            weight_param.shape, generator=generator, dtype=torch.float32
        )
        # The TPU dense loader transposes weights to [input, output]. Use
        # the logical fan-in rather than depending on the stored layout.
        weight *= self.gate.input_size**-0.5
        with torch.no_grad():
            bias_param.copy_(bias)
            weight_param.copy_(weight)

    def process_weights_after_loading(self) -> None:
        self.initialize_dummy_router()
        if self._fuse_tp_up_proj:
            device = self.gate.weight.device
            self.register_buffer(
                "_tp_rank_buf",
                torch.tensor([self.tp_rank], dtype=torch.long, device=device),
                persistent=False,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        sequence_parallel: bool = False,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sequence_parallel or getattr(self, "prefill_group", None) is not None:
            return self._forward_expert_parallel(
                hidden_states, sequence_parallel, padding_mask=padding_mask
            )
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        router_logits, _ = self.gate(hidden_states)
        output = self.experts(hidden_states=hidden_states, router_logits=router_logits)
        if self._fuse_tp_up_proj:
            if self.routed_expert_norm is not None:
                output = self.routed_expert_norm(output)
            assert self.routed_expert_up_proj is not None
            assert self._tp_rank_buf is not None
            output = (
                output.unflatten(-1, (self.tp_size, -1))
                .index_select(-2, self._tp_rank_buf)
                .squeeze(-2)
            )
            output, _ = self.routed_expert_up_proj(output)
            if self.shared_experts is not None:
                output = output + self.shared_experts(hidden_states)
            output = tensor_model_parallel_all_reduce(output)
        return output.view(num_tokens, hidden_size)

    def _forward_expert_parallel(
        self,
        hidden_states: torch.Tensor,
        sequence_parallel: bool,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run EP experts with sharded or replicated tokens, then add shared experts."""
        group = self.prefill_group
        assert group is not None
        router_logits, _ = self.gate(hidden_states)
        shared_group = getattr(self, "shared_group", None)
        down_group = getattr(self.routed_expert_down_proj, "_chip_group", None)
        share_sub_gather = (
            sequence_parallel
            and shared_group is not None
            and down_group is shared_group
            and int(shared_group.world_size) > 1
            and self.shared_experts is not None
        )
        sub_hidden = (
            shared_group.all_gather(hidden_states, dim=0)
            if share_sub_gather
            else None
        )
        latent = hidden_states
        if self.routed_expert_down_proj is not None:
            if isinstance(self.routed_expert_down_proj, IntraChipColumnParallelLinear):
                latent, _ = self.routed_expert_down_proj(
                    latent,
                    sequence_parallel=sequence_parallel,
                    pre_gathered_input=sub_hidden,
                )
            else:
                latent, _ = self.routed_expert_down_proj(latent)
        routed = self.experts.routed_experts
        if isinstance(group, GenericSPCollectives):
            method = cast(
                "VllmCompressedTensorsW4ANMxfp4MoEMethod", routed.quant_method
            )
            weights, ids = moe_routing.route(
                routed,
                latent,
                router_logits,
                topk_weights_float32=method._tpu_topk_weights_float32,
            )
            if padding_mask is not None:
                weights = torch.where(
                    padding_mask[:, None], torch.zeros_like(weights), weights
                )
            moe_chunks = getattr(self, "sp_moe_chunks", 1) if sequence_parallel else 1
            if moe_chunks > 1:
                latent_chunks = latent.reshape(moe_chunks, -1, latent.shape[-1])
                weight_chunks = weights.reshape(moe_chunks, -1, weights.shape[-1])
                id_chunks = ids.reshape(moe_chunks, -1, ids.shape[-1])
                ag_curr = group.gather_moe_inputs(
                    latent_chunks[0],
                    weight_chunks[0],
                    id_chunks[0],
                    num_experts=self.gate.output_size,
                    sequence_parallel=True,
                )
                rs_outputs = []
                for idx in range(moe_chunks):
                    ag_next = None
                    if idx + 1 < moe_chunks:
                        ag_next = group.gather_moe_inputs(
                            latent_chunks[idx + 1],
                            weight_chunks[idx + 1],
                            id_chunks[idx + 1],
                            num_experts=self.gate.output_size,
                            sequence_parallel=True,
                        )
                    out_i = method.apply_with_routing(
                        routed,
                        ag_curr[0],
                        ag_curr[1],
                        ag_curr[2],
                        pipelined=False,
                        padding_already_zeroed=padding_mask is not None,
                    )
                    rs_outputs.append(
                        group.reduce_moe_output(out_i, sequence_parallel=True)
                    )
                    if ag_next is not None:
                        ag_curr = ag_next
                output = torch.cat(rs_outputs, dim=0)
            else:
                latent, weights, ids = group.gather_moe_inputs(
                    latent,
                    weights,
                    ids,
                    num_experts=self.gate.output_size,
                    sequence_parallel=sequence_parallel,
                )
                output = method.apply_with_routing(
                    routed,
                    latent,
                    weights,
                    ids,
                    pipelined=False,
                    padding_already_zeroed=padding_mask is not None,
                )
                output = group.reduce_moe_output(
                    output, sequence_parallel=sequence_parallel
                )
        else:
            if sequence_parallel:
                method = cast(
                    "VllmCompressedTensorsW4ANMxfp4MoEMethod", routed.quant_method
                )
                weights, ids = moe_routing.route(
                    routed,
                    latent,
                    router_logits,
                    topk_weights_float32=method._tpu_topk_weights_float32,
                )
                latent, weights, ids = group.gather_moe_inputs(
                    latent, weights, ids, num_experts=self.gate.output_size
                )
                output = method.apply_with_routing(routed, latent, weights, ids)
            else:
                output = routed.quant_method.apply_monolithic(
                    routed, latent, router_logits
                )
            output = (
                group.reduce_scatter(output, dim=0)
                if sequence_parallel
                else group.all_reduce(output)
            )
        if self.routed_expert_norm is not None:
            output = self.routed_expert_norm(output)
        if getattr(self, "_fuse_generic_sp_up_proj", False):
            assert self.routed_expert_up_proj is not None
            assert self.shared_group is not None
            sub_ws = int(self.shared_group.world_size)
            sub_rank = int(self.shared_group.rank_in_group)
            in_part = int(self.routed_expert_up_proj.input_size_per_partition)
            if sequence_parallel:
                up_in = _subgroup_feature_to_token_a2a(
                    output, self.shared_group, in_part
                )
            else:
                up_in = (
                    output.narrow(-1, sub_rank * in_part, in_part)
                    if sub_ws > 1
                    else output
                )
            up_partial = self.routed_expert_up_proj.quant_method.apply(
                self.routed_expert_up_proj, up_in, None
            )
            if self.shared_experts is not None:
                shared_partial = self.shared_experts(
                    hidden_states,
                    sequence_parallel=sequence_parallel,
                    pre_gathered_input=sub_hidden,
                    reduce_output=False,
                )
                combined = up_partial + shared_partial
            else:
                combined = up_partial
            return (
                self.shared_group.reduce_scatter(combined, dim=0)
                if sequence_parallel
                else self.shared_group.all_reduce(combined)
            )
        if self.routed_expert_up_proj is not None:
            if isinstance(self.routed_expert_up_proj, IntraChipColumnParallelLinear):
                output, _ = self.routed_expert_up_proj(
                    output, sequence_parallel=sequence_parallel
                )
            else:
                output, _ = self.routed_expert_up_proj(output)
        if self.shared_experts is not None:
            output = output + self.shared_experts(
                hidden_states,
                sequence_parallel=sequence_parallel,
                pre_gathered_input=sub_hidden,
            )
        return output
