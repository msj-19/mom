import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from megatron import mpu
from megatron.model.activations import get_activation

from einops import rearrange
import triton 
import triton.language as tl

from fla.ops.gla import fused_chunk_gla, chunk_gla, fused_recurrent_gla
# from .mom_triton import chunk_mom_la
from megatron.model.norms import LayerNorm,get_norm
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
import einops

# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tri Dao.
from typing import Optional, Tuple, Union
import torch
from einops import rearrange, repeat


# def rotate_half(x, interleaved=False):
#     if not interleaved:
#         x1, x2 = x.chunk(2, dim=-1)
#         return torch.cat((-x2, x1), dim=-1)
#     else:
#         x1, x2 = x[..., ::2], x[..., 1::2]
#         return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)


# def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
#     """
#     x: (batch_size, seqlen, nheads, headdim)
#     cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
#     """
#     ro_dim = cos.shape[-1] * 2
#     assert ro_dim <= x.shape[-1]
#     cos = repeat(
#         cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
#     sin = repeat(
#         sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
#     return torch.cat(
#         [x[..., :ro_dim] * cos +
#             rotate_half(x[..., :ro_dim], interleaved) * sin, x[..., ro_dim:]],
#         dim=-1,
#     )


# class ApplyRotaryEmb(torch.autograd.Function):
#     @staticmethod
#     def forward(
#         ctx,
#         x,
#         cos,
#         sin,
#         interleaved=False,
#         inplace=False,
#         seqlen_offsets: Union[int, torch.Tensor] = 0,
#         cu_seqlens: Optional[torch.Tensor] = None,
#         max_seqlen: Optional[int] = None,
#     ):
#         out = apply_rotary(
#             x,
#             cos,
#             sin,
#             seqlen_offsets=seqlen_offsets,
#             cu_seqlens=cu_seqlens,
#             max_seqlen=max_seqlen,
#             interleaved=interleaved,
#             inplace=inplace,
#         )
#         if isinstance(seqlen_offsets, int):
#             # Can't save int with save_for_backward
#             ctx.save_for_backward(cos, sin, cu_seqlens)
#             ctx.seqlen_offsets = seqlen_offsets
#         else:
#             ctx.save_for_backward(cos, sin, cu_seqlens, seqlen_offsets)
#             ctx.seqlen_offsets = None
#         ctx.interleaved = interleaved
#         ctx.inplace = inplace
#         ctx.max_seqlen = max_seqlen
#         return out if not inplace else x

#     @staticmethod
#     def backward(ctx, do):
#         seqlen_offsets = ctx.seqlen_offsets
#         if seqlen_offsets is None:
#             cos, sin, cu_seqlens, seqlen_offsets = ctx.saved_tensors
#         else:
#             cos, sin, cu_seqlens = ctx.saved_tensors
#         # TD [2023-09-02]: For some reason Triton (2.0.0.post1) errors with
#         # "[CUDA]: invalid device context", and cloning makes it work. Idk why. Triton 2.1.0 works.
#         if not ctx.interleaved and not ctx.inplace:
#             do = do.clone()
#         dx = apply_rotary(
#             do,
#             cos,
#             sin,
#             seqlen_offsets=seqlen_offsets,
#             cu_seqlens=cu_seqlens,
#             max_seqlen=ctx.max_seqlen,
#             interleaved=ctx.interleaved,
#             inplace=ctx.inplace,
#             conjugate=True,
#         )
#         return dx, None, None, None, None, None, None, None


# def apply_rotary_emb(
#     x,
#     cos,
#     sin,
#     interleaved=False,
#     inplace=False,
#     seqlen_offsets: Union[int, torch.Tensor] = 0,
#     cu_seqlens: Optional[torch.Tensor] = None,
#     max_seqlen: Optional[int] = None,
# ):
#     """
#     Arguments:
#         x: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
#             else (total_seqlen, nheads, headdim)
#         cos, sin: (seqlen_rotary, rotary_dim / 2)
#         interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
#             of 1st half and 2nd half (GPT-NeoX style).
#         inplace: if True, apply rotary embedding in-place.
#         seqlen_offsets: (batch_size,) or int. Each sequence in x is shifted by this amount.
#             Most commonly used in inference when we have KV cache.
#         cu_seqlens: (batch + 1,) or None
#         max_seqlen: int
#     Return:
#         out: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
#             else (total_seqlen, nheads, headdim)
#     rotary_dim must be <= headdim
#     Apply rotary embedding to the first rotary_dim of x.
#     """
#     return ApplyRotaryEmb.apply(
#         x, cos, sin, interleaved, inplace, seqlen_offsets, cu_seqlens, max_seqlen
#     )

# # For backward compatibility
# apply_rotary_emb_func = apply_rotary_emb

# # vo rotary
# class RotaryEmbedding(torch.nn.Module):
#     """
#     The rotary position embeddings from RoFormer_ (Su et. al).
#     A crucial insight from the method is that the query and keys are
#     transformed by rotation matrices which depend on the relative positions.

#     Other implementations are available in the Rotary Transformer repo_ and in
#     GPT-NeoX_, GPT-NeoX was an inspiration

#     .. _RoFormer: https://arxiv.org/abs/2104.09864
#     .. _repo: https://github.com/ZhuiyiTechnology/roformer
#     .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

#     If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
#     A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
#     Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
#     """

#     def __init__(
#         self,
#         dim: int,
#         base=10000.0,
#         interleaved=False,
#         scale_base=None,
#         pos_idx_in_fp32=True,
#         device=None,
#     ):
#         """
#         interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
#             of 1st half and 2nd half (GPT-NeoX style).
#         pos_idx_in_fp32: if True, the position indices [0.0, ..., seqlen - 1] are in fp32,
#             otherwise they might be in lower precision.
#             This option was added because previously (before 2023-07-02), when we construct
#             the position indices, we use the dtype of self.inv_freq. In most cases this would
#             be fp32, but if the model is trained in pure bf16 (not mixed precision), then
#             self.inv_freq would be bf16, and the position indices are also in bf16.
#             Because of the limited precision of bf16 (e.g. 1995.0 is rounded to 2000.0), the
#             embeddings for some positions will coincide.
#             To maintain compatibility with models previously trained in pure bf16,
#             we add this option.
#         """
#         super().__init__()
#         self.dim = dim
#         self.base = float(base)
#         self.pos_idx_in_fp32 = pos_idx_in_fp32
#         # Generate and save the inverse frequency buffer (non trainable)
#         inv_freq = self._compute_inv_freq(device)
#         self.register_buffer("inv_freq", inv_freq, persistent=False)
#         self.interleaved = interleaved
#         self.scale_base = scale_base
#         scale = (
#             (torch.arange(0, dim, 2, device=device,
#              dtype=torch.float32) + 0.4 * dim) / (1.4 * dim)
#             if scale_base is not None
#             else None
#         )
#         self.register_buffer("scale", scale, persistent=False)

#         self._seq_len_cached = 0
#         self._cos_cached = None
#         self._sin_cached = None
#         self._cos_k_cached = None
#         self._sin_k_cached = None

#     def _compute_inv_freq(self, device=None):
#         return 1.0 / (
#             self.base
#             ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
#         )

#     def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
#         # Reset the tables if the sequence length has changed,
#         # if we're on a new device (possibly due to tracing for instance),
#         # or if we're switching from inference mode to training
#         if (
#             seqlen > self._seq_len_cached
#             or self._cos_cached is None
#             or self._cos_cached.device != device
#             or self._cos_cached.dtype != dtype
#             or (self.training and self._cos_cached.is_inference())
#         ):
#             self._seq_len_cached = seqlen
#             # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
#             # And the output of arange can be quite large, so bf16 would lose a lot of precision.
#             # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
#             if self.pos_idx_in_fp32:
#                 t = torch.arange(seqlen, device=device, dtype=torch.float32)
#                 # We want fp32 here as well since inv_freq will be multiplied with t, and the output
#                 # will be large. Having it in bf16 will lose a lot of precision and cause the
#                 # cos & sin output to change significantly.
#                 # We want to recompute self.inv_freq if it was not loaded in fp32
#                 if self.inv_freq.dtype != torch.float32:
#                     inv_freq = self._compute_inv_freq(device=device)
#                 else:
#                     inv_freq = self.inv_freq
#             else:
#                 t = torch.arange(seqlen, device=device,
#                                  dtype=self.inv_freq.dtype)
#                 inv_freq = self.inv_freq
#             # Don't do einsum, it converts fp32 to fp16 under AMP
#             # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
#             freqs = torch.outer(t, inv_freq)

#             self._cos_cached = torch.cos(freqs).to(dtype)
#             self._sin_cached = torch.sin(freqs).to(dtype)

#     def forward(
#         self,
#         v: torch.Tensor,
#         reverse: bool = False,
#         seqlen_offset: Union[int, torch.Tensor] = 0,
#         max_seqlen: Optional[int] = None,
#     ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
#         seqlen = v.shape[1]
#         if max_seqlen is not None:
#             self._update_cos_sin_cache(max_seqlen, device=v.device, dtype=v.dtype)
#         elif isinstance(seqlen_offset, int):
#             self._update_cos_sin_cache(seqlen + seqlen_offset, device=v.device, dtype=v.dtype)

#         if not reverse:
#             v = apply_rotary_emb_func(
#                 v,
#                 self._cos_cached,
#                 self._sin_cached,
#                 interleaved=self.interleaved,
#                 seqlen_offsets=seqlen_offset,
#             )
#         else:
#             v = apply_rotary_emb_func(
#                 v,
#                 self._cos_cached,
#                 -self._sin_cached, #负数补充
#                 interleaved=self.interleaved,
#                 seqlen_offsets=seqlen_offset,
#             )


#         return v


class LLaMAParallelMLP(nn.Module):
    """LLaMA's MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.

    Note: multiple_of is used to compute the hidden dimension of the MLP
    """

    def __init__(
        self,
        neox_args,
        init_method,
        output_layer_init_method,
        parallel_output=False,
        multiple_of=256,
        MOE=False,
        MoE_mp_size=1,
    ):
        super().__init__()

        self.activation_func,is_gated = get_activation(neox_args)
        self.activation_type = neox_args.activation

        self.multiple_of = multiple_of

        # Allow custom intermediate size, e.g. for Mistral
        if neox_args.intermediate_size is not None:
            ff_dim = neox_args.intermediate_size
        else:
            ff_dim = int(2 * neox_args.hidden_size * 4 / 3)
            ff_dim = self.multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)

        self.w1 = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=ff_dim,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            bias=False,
            MOE=MOE,
            MoE_mp_size=MoE_mp_size,
        )
        self.w3 = mpu.ColumnParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size,
            output_size=ff_dim,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            bias=False,
            MOE=MOE,
            MoE_mp_size=MoE_mp_size,
        )
        self.w2 = mpu.RowParallelLinear(
            neox_args=neox_args,
            input_size=ff_dim,
            output_size=neox_args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            parallel_output=parallel_output,
            bias=False,
            MOE=MOE,
            MoE_mp_size=MoE_mp_size,
        )

    def forward(self, hidden_states):
        w1_out, _ = self.w1(hidden_states)
        w3_out, _ = self.w3(hidden_states)
        return self.w2(self.activation_func(w1_out) * w3_out)


from fla.modules import RotaryEmbedding
# from mom_triton import chunk_mom_la
class ParallelmomAttention(nn.Module):
    def __init__(self, neox_args, init_method, output_layer_init_method,):
        super().__init__()
        self.embed_dim = neox_args.hidden_size
        self.num_heads = neox_args.num_attention_heads
        base =  10000.0 #neox_args.base
        self.ratio = 2 #neox_args.ratio 
        self.shared = False #neox_args.shared #bool
        self.top_k = 1 #neox_args.topk

        self.gate_fn = nn.functional.silu
        self.dk = int(self.embed_dim*0.5)
        self.dv = int(self.embed_dim) 

        self.head_dv = self.dv // self.num_heads 
        self.head_dk = self.dk // self.num_heads 

        self.q_proj = mpu.ColumnParallelLinear(neox_args=neox_args,
                                               input_size=self.embed_dim,
                                               output_size=self.dk,
                                               bias=False,
                                               gather_output=True,
                                               init_method=init_method,
                                               skip_bias_add=not True)
        self.k_proj = mpu.ColumnParallelLinear(neox_args=neox_args,
                                               input_size=self.embed_dim,
                                               output_size=self.dk,
                                               bias=False,
                                               gather_output=True,
                                               init_method=init_method,
                                               skip_bias_add=not True)
        self.g_proj = mpu.ColumnParallelLinear(neox_args=neox_args,
                                               input_size=self.embed_dim,
                                               output_size=self.dv,
                                               bias=False,
                                               gather_output=True,
                                               init_method=init_method,
                                               skip_bias_add=not True)
        
        self.out_proj = mpu.RowParallelLinear(
                                            neox_args=neox_args,
                                            input_size=self.dv,
                                            output_size=self.embed_dim,
                                            input_is_parallel=False,
                                            init_method=output_layer_init_method,
                                            skip_bias_add=not False,
                                            bias=False,
                                            parallel_output=False,
                                            )
        self.v_proj = mpu.ColumnParallelLinear(neox_args=neox_args,
                                               input_size=self.embed_dim,
                                               output_size=self.dv,
                                               bias=False,
                                               gather_output=True,
                                               init_method=init_method,
                                               skip_bias_add=not True)

        self.router_weight = nn.Parameter(torch.empty((self.num_heads,self.head_dv,self.ratio)))
        self.group_norm = nn.LayerNorm(self.head_dv, eps=1e-5, elementwise_affine=False)  

        self.d_conv = 2
        self.conv1d = nn.Conv1d(
            in_channels=self.embed_dim,
            out_channels=self.embed_dim,
            bias=False,
            kernel_size=self.d_conv,
            groups=self.embed_dim,
            padding=self.d_conv - 1,
            # **factory_kwargs,
        )
        self.qkrotary = RotaryEmbedding(dim=self.head_dk,base=base, interleaved=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init  as init
        init.kaiming_uniform_(self.router_weight, a=math.sqrt(5))

    def forward(self, x, hidden_states=None, conv_states=None):
        x = x.transpose(0, 1).contiguous()
        b,l,d = x.shape
        x = rearrange(x, 'b l d -> b d l').contiguous()
        if self.training:
            x = causal_conv1d_fn(
                    x=x,
                    weight=einops.rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias.to(self.precision)
                    if self.conv1d.bias is not None
                    else self.conv1d.bias,
                    activation="silu",
                )
        elif conv_states is None:
            conv_states = nn.functional.pad(
                x, (self.d_conv - x.shape[-1], 0)
            )
            x = causal_conv1d_fn(
                    x=x,
                    weight=einops.rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias.to(self.precision)
                    if self.conv1d.bias is not None
                    else self.conv1d.bias,
                    activation="silu",
                )
        else:
            x = causal_conv1d_update(
                    x,
                    conv_states,
                    weight=einops.rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias.to(self.precision)
                    if self.conv1d.bias is not None
                    else self.conv1d.bias,
                    activation="silu",
                )
            x = x
        x = rearrange(x, 'b d l -> b l d').contiguous()
        q,_ = (self.q_proj(x)) #query_q(b l dk)
        q = self.gate_fn(q)
        k,_ = self.k_proj(x) #get k(b l dk)
        v,_ = self.v_proj(x) #b l 2*self.head_dv
        g,_ = self.g_proj(x) #all get b l d
        # if seqlen_offset == None:

        q = rearrange(q, 'b l (h d) -> b h l d', h = self.num_heads).contiguous()
        k = rearrange(k, 'b l (h d) -> b h l d', h = self.num_heads).contiguous()
        v = rearrange(v, 'b l (h d) -> b h l d', h = self.num_heads).contiguous()
        seqlen_offset = 0
        q, k = self.qkrotary(q,k,seqlen_offset) #wo qk_rotary    
        output,new_hidden_states = self.gated_linear_attention(q, k, v,state = hidden_states)

        output = rearrange(output,'b h l d -> b l h d')
        output = self.group_norm(output)

        output = self.gate_fn(g) * (output.view(b,l,d))
        output,_  = self.out_proj(output)
        output = output.transpose(0, 1)
        return output,new_hidden_states,conv_states

    def gated_linear_attention(self,q, k, v, state=None):

        '''torch qk version'''
        b,h,l,d = v.shape #b h l d 
        dk = q.shape[-1] # h d r
        logits = torch.matmul(v,self.router_weight)#get b h l r'
        
        scores = logits.softmax(dim=-1)
        topk_score , topk_idx = torch.topk(scores,k = self.top_k,dim=-1,sorted=False)#get b,h,l,top_k
        if self.top_k>1:#norm 
            sum_score = topk_score.sum(dim=-1,keepdim=True)+1e-20
            topk_score = topk_score/sum_score

        # o_moe,f_state = chunk_mom_la(q, k, v, topk_score,topk_idx,self.ratio, -1, initial_state=None, output_final_state=True)

        masked_scores = torch.zeros_like(scores,device=q.device)
        masked_scores.scatter_(-1, topk_idx, topk_score)
        masked_indices = torch.zeros([b,h,l,self.ratio],dtype=torch.int64,device='cuda')
        masked_indices = torch.where(masked_scores==0,0,1)
        sum_indices = torch.cumsum(masked_indices,dim=2)#.unsqueeze(-1)#return b h l r
        norm_score = torch.where(sum_indices==0,0,1/sum_indices).unsqueeze(-1)

        q_exp = torch.einsum('b h l d, b h l r-> b h l r d',q,masked_scores)
        k_exp = torch.einsum('b h l d, b h l r-> b h l r d',k,masked_indices)

        q_exp = q_exp*norm_score
        q_exp = rearrange(q_exp,'b h l r d -> b h l (r d)').to(q)
        k_exp = rearrange(k_exp,'b h l r d -> b h l (r d)').to(q)

        qk = q_exp @ k_exp.transpose(-1,-2) * ((dk**-0.5))
        mask = (torch.ones(l,l, dtype=torch.bool, device=q.device).tril(diagonal=0))
        qk = (qk*mask.to(qk)).to(q)
        o_moe = qk@v
        
        return o_moe, None




class ParallelmomResidualLayer(nn.Module):
    def __init__(
        self,
        neox_args,
        init_method,
        output_layer_init_method,
        layer_number,
        use_cache=False
    ):
        super().__init__()
        
        assert not use_cache, "gla layer's use_cache is not implemented"
        self.neox_args = neox_args
        self.layer_number = layer_number

        self.hidden_states = None
        self.conv_states = None
        
        norm, eps = get_norm(neox_args)
        self.input_layernorm = norm(neox_args.hidden_size, eps=eps)
        self.post_attention_layernorm = norm(neox_args.hidden_size, eps=eps)
        self.mlp = LLaMAParallelMLP(
            neox_args=neox_args,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method
        )
        self.hidden_dropout = neox_args.hidden_dropout

        self.attention = ParallelmomAttention(neox_args, 
                                                init_method, 
                                                output_layer_init_method)

        
        self.mlp = LLaMAParallelMLP(
                    neox_args=neox_args,
                    init_method=init_method,
                    output_layer_init_method=output_layer_init_method,
                )
    
    def clear_cache(self):
        self.hidden_states = None
        self.conv_states = None
        
    def forward(self, x, attention_mask, layer_past=None):
        residual = x  #(2048,16,1024) (l,b,d)
        
        moe_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # attention_output,_ = self.attention(self.input_layernorm(x))
        if self.training:
            attention_output, _, _ = self.attention(self.input_layernorm(x))
        else:
            attention_output, self.hidden_states, self.conv_states = self.attention(self.input_layernorm(x), hidden_states=self.hidden_states, conv_states=self.conv_states)
        
        with torch.enable_grad():
            attention_output = (
                        torch.nn.functional.dropout(
                            attention_output,
                            p=self.hidden_dropout,
                            training=self.training,
                        )
                        + residual
                    )
            
        layernorm_output = self.post_attention_layernorm(attention_output)
        
        mlp_output, _ = self.mlp(layernorm_output)
        
        with torch.enable_grad():
            output = mlp_output + attention_output
            
        return output, moe_loss

  
class ParallelmomResidualLayerPipe(ParallelmomResidualLayer):
    """Extends ParallelTransformerLayer to forward attention_mask through the pipeline."""

    def forward(self, args):
        assert (
            len(args) == 2
        ), "ParallelTransformerLayerPipe expects 2 arguments - hidden_states and attention_mask"
        hidden_states, attention_mask = args
        # we are returning just [hidden_states, mask]
        output, moe_loss = super().forward(hidden_states, attention_mask)
        # auxiliary output
        self.last_moe_loss = moe_loss
        return output, attention_mask
    

##### [Note] clear cache before each batch!!! #####
# for module in model.modules():
#     if hasattr(module, 'clear_cache'):
#         module.clear_cache()
