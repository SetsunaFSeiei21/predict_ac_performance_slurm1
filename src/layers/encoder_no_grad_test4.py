# import math
# from typing import Optional, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class HeadwiseTrueGradientProxyKeyAttention(nn.Module):
#     """
#     Head-wise true-gradient / proxy-gradient attention.

#     Forward during training:
#         Q = sg(H) Wq
#         K = sg(H) Wk diag(sg(alpha))
#         V = H Wv
#         O = softmax(QK^T / sqrt(d) + mask) V Wo

#     Backward rule for Wk:
#         For selected true-gradient heads:
#             grad(Wk_h) = real score-gradient

#         For remaining proxy-gradient heads:
#             grad(Wk_h) = value-gradient proxy

#     Backward rule for alpha / D_alpha:
#         alpha does not receive score-path gradients because it is detached
#         in the K/score path.

#         For proxy-gradient heads, alpha receives gradients through the
#         proxy value-gradient path.

#         For true-gradient heads, alpha is effectively fixed unless another
#         path explicitly updates it.

#     Main purpose:
#         Only a subset of heads keep the real score-gradient graph.
#         The remaining heads use a diagonal value-gradient proxy to reduce
#         backward graph and saved-tensor overhead.

#     Inference:
#         Uses a fast standard Q/K/V projection path to avoid head-wise
#         Python loops and proxy-gradient construction.
#     """

#     def __init__(
#         self,
#         embed_dim: int,
#         num_heads: int,
#         dropout: float = 0.0,
#         bias: bool = True,
#         init_alpha: float = 1.0,
#         alpha_eps: float = 1e-6,
#         true_head_ratio: float = 0.5,
#         need_attn_score: bool = False,
#     ) -> None:
#         super().__init__()

#         assert embed_dim % num_heads == 0, (
#             f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
#         )
#         assert init_alpha > 0, "init_alpha must be positive."
#         assert 0.0 <= true_head_ratio <= 1.0, (
#             f"true_head_ratio must be in [0, 1], got {true_head_ratio}"
#         )

#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         self.dropout = dropout
#         self.alpha_eps = alpha_eps
#         self.true_head_ratio = float(true_head_ratio)
#         self.need_attn_score = need_attn_score

#         self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
#         self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
#         self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
#         self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

#         # alpha_c = softplus(log_alpha_c) + eps
#         target = max(init_alpha - alpha_eps, 1e-6)
#         self._init_log_alpha_value = math.log(math.expm1(target))

#         self.log_alpha = nn.Parameter(
#             torch.full(
#                 size=(embed_dim,),
#                 fill_value=self._init_log_alpha_value,
#                 dtype=torch.float32,
#             )
#         )

#         true_head_num = int(round(num_heads * true_head_ratio))
#         true_head_num = max(0, min(num_heads, true_head_num))

#         true_head_mask = torch.zeros(num_heads, dtype=torch.bool)
#         if true_head_num > 0:
#             true_head_mask[:true_head_num] = True

#         # Shape: [num_heads]
#         self.register_buffer("true_head_mask", true_head_mask)

#         # Shape: [embed_dim]
#         true_channel_mask = true_head_mask.repeat_interleave(self.head_dim)
#         self.register_buffer("true_channel_mask", true_channel_mask)

#         self.reset_parameters()

#     @property
#     def alpha(self) -> torch.Tensor:
#         """
#         Shape:
#             [embed_dim]
#         """
#         return F.softplus(self.log_alpha) + self.alpha_eps

#     @property
#     def true_head_num(self) -> int:
#         return int(self.true_head_mask.sum().item())

#     @property
#     def proxy_head_num(self) -> int:
#         return int((~self.true_head_mask).sum().item())

#     def reset_parameters(self) -> None:
#         nn.init.xavier_uniform_(self.q_proj.weight)
#         nn.init.xavier_uniform_(self.k_proj.weight)
#         nn.init.xavier_uniform_(self.v_proj.weight)
#         nn.init.xavier_uniform_(self.out_proj.weight)

#         if self.q_proj.bias is not None:
#             nn.init.zeros_(self.q_proj.bias)
#         if self.k_proj.bias is not None:
#             nn.init.zeros_(self.k_proj.bias)
#         if self.v_proj.bias is not None:
#             nn.init.zeros_(self.v_proj.bias)
#         if self.out_proj.bias is not None:
#             nn.init.zeros_(self.out_proj.bias)

#         with torch.no_grad():
#             self.log_alpha.fill_(self._init_log_alpha_value)

#         # Strictly freeze Wq/bq.
#         for param in self.q_proj.parameters():
#             param.requires_grad_(False)

#         self._save_initial_q_state()

#     def _save_initial_q_state(self) -> None:
#         with torch.no_grad():
#             q_weight = self.q_proj.weight.detach().clone()

#             if "_initial_q_weight" in self._buffers:
#                 self._buffers["_initial_q_weight"] = q_weight
#             else:
#                 self.register_buffer("_initial_q_weight", q_weight)

#             if self.q_proj.bias is not None:
#                 q_bias = self.q_proj.bias.detach().clone()

#                 if "_initial_q_bias" in self._buffers:
#                     self._buffers["_initial_q_bias"] = q_bias
#                 else:
#                     self.register_buffer("_initial_q_bias", q_bias)
#             else:
#                 if "_initial_q_bias" not in self._buffers:
#                     self.register_buffer("_initial_q_bias", None)

#     @torch.no_grad()
#     def restore_frozen_q(self) -> None:
#         self.q_proj.weight.copy_(
#             self._initial_q_weight.to(self.q_proj.weight.device)
#         )

#         if self.q_proj.bias is not None and self._initial_q_bias is not None:
#             self.q_proj.bias.copy_(
#                 self._initial_q_bias.to(self.q_proj.bias.device)
#             )

#     @torch.no_grad()
#     def check_q_change(self) -> tuple[float, float]:
#         weight_diff_max = (
#             self.q_proj.weight
#             - self._initial_q_weight.to(self.q_proj.weight.device)
#         ).abs().max().item()

#         if self.q_proj.bias is not None and self._initial_q_bias is not None:
#             bias_diff_max = (
#                 self.q_proj.bias
#                 - self._initial_q_bias.to(self.q_proj.bias.device)
#             ).abs().max().item()
#         else:
#             bias_diff_max = 0.0

#         return weight_diff_max, bias_diff_max

#     def _shape_to_heads(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         [B, N, D] -> [B, H, N, Dh]
#         """
#         B, N, D = x.shape

#         return (
#             x.view(B, N, self.num_heads, self.head_dim)
#             .transpose(1, 2)
#             .contiguous()
#         )

#     def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         [B, H, N, Dh] -> [B, N, D]
#         """
#         B, H, N, Dh = x.shape

#         return (
#             x.transpose(1, 2)
#             .contiguous()
#             .view(B, N, H * Dh)
#         )

#     def _build_sdpa_mask(
#         self,
#         attention_mask: Optional[torch.Tensor],
#         x: torch.Tensor,
#     ) -> Optional[torch.Tensor]:
#         """
#         Compatible with torch.nn.functional.scaled_dot_product_attention.

#         Supported input:
#             None
#             [N, N]
#             [B, N, N]

#         Mask convention:
#             additive mask: 0 allowed, negative value blocked
#             binary mask:   1 allowed, 0 blocked
#         """
#         if attention_mask is None:
#             return None

#         mask = attention_mask.to(device=x.device)

#         # Additive mask: 0 / -1e9
#         if torch.is_floating_point(mask) and torch.min(mask) < 0:
#             if mask.dim() == 2:
#                 return mask.to(dtype=x.dtype)

#             if mask.dim() == 3:
#                 return mask.unsqueeze(1).to(dtype=x.dtype)

#             raise ValueError(
#                 f"Unsupported additive attention_mask shape: {attention_mask.shape}. "
#                 "Expected [N, N] or [B, N, N]."
#             )

#         # Binary mask: 1 allowed, 0 blocked.
#         if mask.dim() == 2:
#             N = mask.shape[0]
#             eye = torch.eye(N, device=mask.device, dtype=mask.dtype)
#             mask = torch.maximum(mask, eye)
#             additive_mask = (1.0 - mask.float()) * -1e9
#             return additive_mask.to(dtype=x.dtype)

#         if mask.dim() == 3:
#             B, N, _ = mask.shape
#             eye = torch.eye(N, device=mask.device, dtype=mask.dtype).unsqueeze(0)
#             mask = torch.maximum(mask, eye)
#             additive_mask = (1.0 - mask.float()) * -1e9
#             return additive_mask.unsqueeze(1).to(dtype=x.dtype)

#         raise ValueError(
#             f"Unsupported attention_mask shape: {attention_mask.shape}. "
#             "Expected [N, N] or [B, N, N]."
#         )

#     def _manual_attention_with_score(
#         self,
#         q: torch.Tensor,
#         k: torch.Tensor,
#         v: torch.Tensor,
#         attn_mask: Optional[torch.Tensor],
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

#         if attn_mask is not None:
#             score = score + attn_mask

#         attn_score = torch.softmax(score, dim=-1)

#         if self.training and self.dropout > 0:
#             attn_score = F.dropout(attn_score, p=self.dropout, training=True)

#         out = torch.matmul(attn_score, v)

#         return out, attn_score

#     def _compute_k_by_heads(self, x_qk: torch.Tensor) -> torch.Tensor:
#         """
#         Compute K head by head.

#         For true-gradient heads:
#             K_h = x_qk @ Wk_h^T + b_h
#             with autograd enabled for Wk_h.

#         For proxy heads:
#             K_h is computed under torch.no_grad(),
#             so Wk_h does not receive score-gradient.

#         Returns:
#             k_raw: [B, N, D]
#         """
#         k_weight = self.k_proj.weight
#         k_bias = self.k_proj.bias

#         k_heads = []

#         for h in range(self.num_heads):
#             start = h * self.head_dim
#             end = (h + 1) * self.head_dim

#             weight_h = k_weight[start:end, :]
#             bias_h = k_bias[start:end] if k_bias is not None else None

#             if bool(self.true_head_mask[h].item()):
#                 # This head keeps real score-gradient for Wk_h.
#                 k_h = F.linear(x_qk, weight_h, bias_h)
#             else:
#                 # This head only participates in forward.
#                 # Its Wk_h update comes from proxy-gradient through V path.
#                 with torch.no_grad():
#                     k_h = F.linear(x_qk, weight_h, bias_h)

#             k_heads.append(k_h)

#         k_raw = torch.cat(k_heads, dim=-1)

#         return k_raw

#     def forward(
#         self,
#         x: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
#         """
#         x:
#             [B, N, D]
#         """
#         alpha_channel = self.alpha.to(device=x.device, dtype=x.dtype)

#         # ==========================================================
#         # Fast inference path
#         # ==========================================================
#         # In eval mode, proxy-gradient construction is unnecessary.
#         # Use one-shot Q/K/V projections to avoid head-wise Python loops.
#         if not self.training:
#             x_qk = x.detach()

#             q = self.q_proj(x_qk)
#             k = self.k_proj(x_qk) * alpha_channel.view(1, 1, -1)
#             v = self.v_proj(x)

#             q = self._shape_to_heads(q)
#             k = self._shape_to_heads(k)
#             v = self._shape_to_heads(v)

#             sdpa_mask = self._build_sdpa_mask(attention_mask, x)

#             if self.need_attn_score:
#                 out, attn_score = self._manual_attention_with_score(
#                     q=q,
#                     k=k,
#                     v=v,
#                     attn_mask=sdpa_mask,
#                 )
#             else:
#                 out = F.scaled_dot_product_attention(
#                     q,
#                     k,
#                     v,
#                     attn_mask=sdpa_mask,
#                     dropout_p=0.0,
#                     is_causal=False,
#                 )
#                 attn_score = None

#             out = self._merge_heads(out)
#             out = self.out_proj(out)

#             return out, attn_score

#         # ==========================================================
#         # Training path
#         # ==========================================================
#         true_channel_mask = self.true_channel_mask.to(device=x.device)
#         proxy_channel_mask = (~true_channel_mask).to(dtype=x.dtype)

#         # Q/K use detached hidden states, so Q/K paths do not send gradients to H.
#         x_qk = x.detach()

#         # Q is frozen and no graph is needed.
#         with torch.no_grad():
#             q = self.q_proj(x_qk)

#         # K:
#         #   true heads: keep Wk score-gradient
#         #   proxy heads: no Wk score-gradient
#         k_raw = self._compute_k_by_heads(x_qk)

#         # Channel-wise alpha participates in K forward,
#         # but does not receive score-path gradients.
#         #
#         # Therefore D_alpha is not learned through the score path.
#         k = k_raw * alpha_channel.detach().view(1, 1, -1)

#         # V path with diagonal proxy gradient for proxy heads only.
#         #
#         # Forward:
#         #   proxy_weight_term - proxy_weight_term.detach() == 0
#         #   so v_weight_eff == v_proj.weight
#         #
#         # Backward:
#         #   grad(Wv) normal;
#         #   grad(Wk rows of proxy heads) receives alpha_c * grad(Wv_c);
#         #   grad(alpha_c of proxy heads) receives proxy value-gradient;
#         #   true heads receive no proxy-gradient.
#         #
#         # This makes D_alpha learn through the proxy value-gradient path,
#         # rather than the score path.
#         proxy_row_scale = (proxy_channel_mask * alpha_channel).view(-1, 1)

#         proxy_weight_term = proxy_row_scale * self.k_proj.weight

#         v_weight_eff = self.v_proj.weight + (
#             proxy_weight_term - proxy_weight_term.detach()
#         )

#         if self.v_proj.bias is not None:
#             proxy_bias_term = proxy_channel_mask * alpha_channel * self.k_proj.bias
#             v_bias_eff = self.v_proj.bias + (
#                 proxy_bias_term - proxy_bias_term.detach()
#             )
#         else:
#             v_bias_eff = None

#         v = F.linear(x, v_weight_eff, v_bias_eff)

#         q = self._shape_to_heads(q)
#         k = self._shape_to_heads(k)
#         v = self._shape_to_heads(v)

#         sdpa_mask = self._build_sdpa_mask(attention_mask, x)

#         if self.need_attn_score:
#             out, attn_score = self._manual_attention_with_score(
#                 q=q,
#                 k=k,
#                 v=v,
#                 attn_mask=sdpa_mask,
#             )
#         else:
#             out = F.scaled_dot_product_attention(
#                 q,
#                 k,
#                 v,
#                 attn_mask=sdpa_mask,
#                 dropout_p=self.dropout if self.training else 0.0,
#                 is_causal=False,
#             )
#             attn_score = None

#         out = self._merge_heads(out)
#         out = self.out_proj(out)

#         return out, attn_score


# class Encoder_No_Grad_Test(nn.Module):
#     """
#     Encoder block with head-wise true-gradient / proxy-gradient attention.

#     Block-level structure is aligned with the standard Encoder:

#         Pre-LN -> Attention -> residual
#         Pre-LN -> FFN       -> residual projection

#     Main approximation:
#         Some heads keep real Wk score-gradient.
#         Other heads use diagonal value-gradient proxy.

#     Removed:
#         Q/K score-gradient-to-hidden path
#         Wq update

#     Preserved:
#         real Wk score-gradient for selected heads
#         proxy Wk update for remaining heads
#         V/content-gradient-to-hidden path
#         out projection / FFN / LayerNorm
#     """

#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dim: int,
#         output_dim: int,
#         num_heads: int,
#         dropout: float,
#         init_alpha: float = 1.0,
#         true_head_ratio: float = 0.5,
#         need_attn_score: bool = False,
#         **kwargs,
#     ) -> None:
#         super().__init__()

#         assert input_dim % num_heads == 0, (
#             f"input_dim={input_dim} must be divisible by num_heads={num_heads}"
#         )

#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.num_heads = num_heads
#         self.need_attn_score = need_attn_score
#         self.true_head_ratio = float(true_head_ratio)

#         self.attention = HeadwiseTrueGradientProxyKeyAttention(
#             embed_dim=input_dim,
#             num_heads=num_heads,
#             dropout=dropout,
#             bias=True,
#             init_alpha=init_alpha,
#             true_head_ratio=true_head_ratio,
#             need_attn_score=need_attn_score,
#         )

#         self.normalize_layer1 = nn.LayerNorm(normalized_shape=input_dim)
#         self.normalize_layer2 = nn.LayerNorm(normalized_shape=input_dim)

#         self.ffn = nn.Sequential(
#             nn.Linear(in_features=input_dim, out_features=hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(in_features=hidden_dim, out_features=output_dim),
#             nn.Dropout(dropout),
#         )

#         # Align with the standard Encoder.
#         # If output_dim != input_dim, project attention residual to output_dim.
#         self.residual_proj = (
#             nn.Identity()
#             if output_dim == input_dim
#             else nn.Linear(input_dim, output_dim)
#         )

#         self._init_weight()

#     def forward(
#         self,
#         x: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
#         """
#         x:
#             [batch_size, node_num, input_dim]
#         """

#         # Pre-LayerNorm + no-grad/proxy-gradient attention
#         normed_x = self.normalize_layer1(x)

#         attn_out, attn_score = self.attention(
#             x=normed_x,
#             attention_mask=attention_mask,
#         )

#         # Attention residual
#         attn_out = attn_out + x

#         # Pre-LayerNorm + FFN
#         normed_attn_out = self.normalize_layer2(attn_out)
#         ffn_out = self.ffn(normed_attn_out)

#         # FFN residual
#         x = ffn_out + self.residual_proj(attn_out)

#         return x, attn_score

#     def restore_frozen_q(self) -> None:
#         self.attention.restore_frozen_q()

#     def check_q_change(self) -> tuple[float, float]:
#         return self.attention.check_q_change()

#     def get_alpha(self) -> list[float]:
#         return self.attention.alpha.detach().cpu().tolist()

#     def get_true_head_ratio(self) -> float:
#         return self.true_head_ratio

#     def get_true_head_num(self) -> int:
#         return self.attention.true_head_num

#     def get_proxy_head_num(self) -> int:
#         return self.attention.proxy_head_num

#     def _init_weight(self) -> None:
#         # Initialize attention.
#         self.attention.reset_parameters()

#         # Initialize FFN.
#         ffn_linear_index = [
#             idx for idx, sub_layer in enumerate(self.ffn)
#             if isinstance(sub_layer, nn.Linear)
#         ]

#         for idx, sub_layer in enumerate(self.ffn):
#             if isinstance(sub_layer, nn.Linear):
#                 if idx != ffn_linear_index[-1]:
#                     nn.init.kaiming_uniform_(sub_layer.weight, nonlinearity="relu")
#                 else:
#                     nn.init.kaiming_uniform_(sub_layer.weight, mode="fan_in")

#                 if sub_layer.bias is not None:
#                     nn.init.zeros_(sub_layer.bias)

#         # Initialize residual projection.
#         if isinstance(self.residual_proj, nn.Linear):
#             nn.init.xavier_uniform_(self.residual_proj.weight)

#             if self.residual_proj.bias is not None:
#                 nn.init.zeros_(self.residual_proj.bias)

#         # Initialize LayerNorm.
#         nn.init.constant_(self.normalize_layer1.weight, 1.0)
#         nn.init.constant_(self.normalize_layer1.bias, 0.0)
#         nn.init.constant_(self.normalize_layer2.weight, 1.0)
#         nn.init.constant_(self.normalize_layer2.bias, 0.0)
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ProxyKeyAttentionFunction(torch.autograd.Function):
    """
    Custom-backward implementation for head-wise true-gradient / proxy-gradient attention.

    Forward:
        x_qk = detach(x)
        Q = x_qk Wq^T + bq
        K_raw = x_qk Wk^T + bk
        K = K_raw * alpha
        V = x Wv^T + bv
        O = softmax(QK^T / sqrt(d) + mask) V

    Backward:
        grad_x:
            only from V path

        grad_Wq:
            None, because Wq is frozen

        grad_Wk:
            true heads:
                real score-gradient from attention score path

            proxy heads:
                proxy-gradient from value path:
                    grad_Wk_proxy = alpha * grad_Wv_proxy

        grad_alpha:
            only proxy heads receive proxy value-gradient:
                    grad_alpha_proxy = <grad_Wv_proxy, Wk_proxy>

            score-path alpha gradient is intentionally not used.
    """

    @staticmethod
    def _shape_to_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
        """
        [B, N, D] -> [B, H, N, Dh]
        """
        B, N, D = x.shape
        head_dim = D // num_heads
        return (
            x.view(B, N, num_heads, head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    @staticmethod
    def _merge_heads(x: torch.Tensor) -> torch.Tensor:
        """
        [B, H, N, Dh] -> [B, N, D]
        """
        B, H, N, Dh = x.shape
        return (
            x.transpose(1, 2)
            .contiguous()
            .view(B, N, H * Dh)
        )

    @staticmethod
    def _merge_true_heads(x: torch.Tensor) -> torch.Tensor:
        """
        [B, T, N, Dh] -> [B, N, T * Dh]
        """
        B, T, N, Dh = x.shape
        return (
            x.transpose(1, 2)
            .contiguous()
            .view(B, N, T * Dh)
        )

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        q_weight: torch.Tensor,
        q_bias: Optional[torch.Tensor],
        k_weight: torch.Tensor,
        k_bias: Optional[torch.Tensor],
        v_weight: torch.Tensor,
        v_bias: Optional[torch.Tensor],
        log_alpha: torch.Tensor,
        additive_mask: Optional[torch.Tensor],
        num_heads: int,
        true_head_num: int,
        dropout_p: float,
        training: bool,
        alpha_eps: float,
    ) -> torch.Tensor:
        """
        x:
            [B, N, D]

        q_weight / k_weight / v_weight:
            [D, D]

        additive_mask:
            None, [1, 1, N, N], [B, 1, N, N], or [B, H, N, N]
        """
        B, N, D = x.shape
        head_dim = D // num_heads
        true_head_num = int(true_head_num)
        true_dim = true_head_num * head_dim
        scale = 1.0 / math.sqrt(head_dim)

        alpha = F.softplus(log_alpha) + alpha_eps
        alpha = alpha.to(device=x.device, dtype=x.dtype)

        x_qk = x.detach()

        # Q is frozen.
        q = F.linear(x_qk, q_weight, q_bias)

        # K is computed in one shot.
        # Although alpha participates in forward score computation, its gradient
        # is manually defined in backward and does not come from score path.
        k_raw = F.linear(x_qk, k_weight, k_bias)
        k = k_raw * alpha.view(1, 1, D)

        # V keeps content-gradient-to-hidden path.
        v = F.linear(x, v_weight, v_bias)

        q_h = _ProxyKeyAttentionFunction._shape_to_heads(q, num_heads)
        k_h = _ProxyKeyAttentionFunction._shape_to_heads(k, num_heads)
        v_h = _ProxyKeyAttentionFunction._shape_to_heads(v, num_heads)

        score = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale

        if additive_mask is not None:
            score = score + additive_mask.to(device=x.device, dtype=x.dtype)

        attn_score = torch.softmax(score, dim=-1)

        if training and dropout_p > 0.0:
            keep_prob = 1.0 - dropout_p
            dropout_mask = (
                torch.rand_like(attn_score) < keep_prob
            ).to(attn_score.dtype) / keep_prob
            attn_used = attn_score * dropout_mask
        else:
            dropout_mask = torch.empty(0, device=x.device, dtype=x.dtype)
            attn_used = attn_score

        out_h = torch.matmul(attn_used, v_h)
        out = _ProxyKeyAttentionFunction._merge_heads(out_h)

        # Save only true-head Q/K_raw for score-gradient.
        # Proxy heads do not need Q/K_raw score-gradient graph.
        if true_head_num > 0:
            q_true = q_h[:, :true_head_num, :, :].contiguous()
            k_raw_true = k_raw[:, :, :true_dim].contiguous()
        else:
            q_true = torch.empty(
                B, 0, N, head_dim, device=x.device, dtype=x.dtype
            )
            k_raw_true = torch.empty(
                B, N, 0, device=x.device, dtype=x.dtype
            )

        if k_bias is None:
            k_bias_saved = torch.empty(0, device=x.device, dtype=x.dtype)
        else:
            k_bias_saved = k_bias

        ctx.save_for_backward(
            x,
            q_true,
            k_raw_true,
            v_h,
            attn_score,
            dropout_mask,
            alpha,
            log_alpha,
            v_weight,
            k_weight,
            k_bias_saved,
        )

        ctx.num_heads = int(num_heads)
        ctx.true_head_num = int(true_head_num)
        ctx.head_dim = int(head_dim)
        ctx.true_dim = int(true_dim)
        ctx.scale = float(scale)
        ctx.dropout_p = float(dropout_p)
        ctx.training = bool(training)
        ctx.has_k_bias = k_bias is not None
        ctx.has_v_bias = v_bias is not None

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        grad_out:
            [B, N, D]
        """
        (
            x,
            q_true,
            k_raw_true,
            v_h,
            attn_score,
            dropout_mask,
            alpha,
            log_alpha,
            v_weight,
            k_weight,
            k_bias_saved,
        ) = ctx.saved_tensors

        num_heads = ctx.num_heads
        true_head_num = ctx.true_head_num
        true_dim = ctx.true_dim
        scale = ctx.scale
        dropout_p = ctx.dropout_p
        training = ctx.training

        B, N, D = x.shape

        grad_out_h = _ProxyKeyAttentionFunction._shape_to_heads(
            grad_out.contiguous(),
            num_heads,
        )

        if training and dropout_p > 0.0:
            attn_used = attn_score * dropout_mask
        else:
            attn_used = attn_score

        # ==========================================================
        # 1. All heads need grad_V because V/content path is preserved.
        # ==========================================================
        grad_v_h = torch.matmul(
            attn_used.transpose(-2, -1),
            grad_out_h,
        )

        grad_v = _ProxyKeyAttentionFunction._merge_heads(grad_v_h)

        x_flat = x.reshape(B * N, D)
        grad_v_flat = grad_v.reshape(B * N, D)

        grad_v_weight = grad_v_flat.transpose(0, 1).matmul(x_flat)
        grad_v_bias = grad_v_flat.sum(dim=0)

        # Hidden gradient only comes from V path.
        grad_x = grad_v_flat.matmul(v_weight).view(B, N, D)

        # ==========================================================
        # 2. Initialize gradients for K and alpha.
        # ==========================================================
        grad_k_weight = torch.zeros_like(k_weight)
        grad_k_bias = torch.zeros(D, device=x.device, dtype=x.dtype)
        grad_log_alpha = torch.zeros_like(log_alpha)

        # ==========================================================
        # 3. True heads: compute real score-gradient for Wk only.
        #    Alpha does not receive score-path gradient.
        # ==========================================================
        if true_head_num > 0:
            grad_out_true = grad_out_h[:, :true_head_num, :, :]
            v_true = v_h[:, :true_head_num, :, :]
            attn_true = attn_score[:, :true_head_num, :, :]

            if training and dropout_p > 0.0:
                dropout_true = dropout_mask[:, :true_head_num, :, :]
                attn_used_true = attn_true * dropout_true
            else:
                dropout_true = None
                attn_used_true = attn_true

            # dA_used = dO V^T
            grad_attn_used_true = torch.matmul(
                grad_out_true,
                v_true.transpose(-2, -1),
            )

            # Dropout backward.
            if training and dropout_p > 0.0:
                grad_attn_true = grad_attn_used_true * dropout_true
            else:
                grad_attn_true = grad_attn_used_true

            # A = softmax(S)
            softmax_dot_true = (
                grad_attn_true * attn_true
            ).sum(dim=-1, keepdim=True)

            grad_score_true = attn_true * (
                grad_attn_true - softmax_dot_true
            )

            # S = Q K^T / sqrt(Dh)
            grad_k_true_h = torch.matmul(
                grad_score_true.transpose(-2, -1),
                q_true,
            ) * scale

            grad_k_true = _ProxyKeyAttentionFunction._merge_true_heads(
                grad_k_true_h
            )

            # K = K_raw * alpha
            # Alpha is treated as stop-gradient for score path.
            alpha_true = alpha[:true_dim].view(1, 1, true_dim)
            grad_k_raw_true = grad_k_true * alpha_true

            x_qk_flat = x.detach().reshape(B * N, D)
            grad_k_raw_true_flat = grad_k_raw_true.reshape(B * N, true_dim)

            grad_k_weight_true = grad_k_raw_true_flat.transpose(0, 1).matmul(
                x_qk_flat
            )
            grad_k_bias_true = grad_k_raw_true_flat.sum(dim=0)

            grad_k_weight[:true_dim, :] = grad_k_weight_true
            grad_k_bias[:true_dim] = grad_k_bias_true

        # ==========================================================
        # 4. Proxy heads: skip score-gradient.
        #    Inject proxy-gradient from value path.
        #
        # Effective proxy term:
        #     Wv_eff = Wv + (alpha * Wk - sg(alpha * Wk))
        #
        # Therefore:
        #     dL/dWk_proxy    = alpha * dL/dWv_proxy
        #     dL/dalpha_proxy = <dL/dWv_proxy, Wk_proxy>
        # ==========================================================
        if true_dim < D:
            alpha_proxy = alpha[true_dim:]

            # Wk proxy gradient.
            grad_k_weight[true_dim:, :] = (
                alpha_proxy.view(-1, 1)
                * grad_v_weight[true_dim:, :]
            )

            if ctx.has_k_bias and ctx.has_v_bias:
                grad_k_bias[true_dim:] = (
                    alpha_proxy
                    * grad_v_bias[true_dim:]
                )

            # Alpha proxy gradient from weight proxy path.
            grad_alpha_proxy = (
                grad_v_weight[true_dim:, :]
                * k_weight[true_dim:, :]
            ).sum(dim=1)

            # Optional bias contribution:
            # v_bias_eff = v_bias + alpha * k_bias - sg(alpha * k_bias)
            if ctx.has_k_bias and ctx.has_v_bias:
                grad_alpha_proxy = grad_alpha_proxy + (
                    grad_v_bias[true_dim:]
                    * k_bias_saved[true_dim:]
                )

            grad_log_alpha_proxy = grad_alpha_proxy * torch.sigmoid(
                log_alpha[true_dim:]
            )

            grad_log_alpha[true_dim:] = grad_log_alpha_proxy.to(
                dtype=log_alpha.dtype
            )

        grad_q_weight = None
        grad_q_bias = None

        if not ctx.has_k_bias:
            grad_k_bias = None

        if not ctx.has_v_bias:
            grad_v_bias = None

        return (
            grad_x,
            grad_q_weight,
            grad_q_bias,
            grad_k_weight,
            grad_k_bias,
            grad_v_weight,
            grad_v_bias,
            grad_log_alpha,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class HeadwiseTrueGradientProxyKeyAttention(nn.Module):
    """
    Optimized custom-backward head-wise true-gradient / proxy-gradient attention.

    Training path:
        Uses _ProxyKeyAttentionFunction.

    Eval path:
        Uses normal fast Q/K/V projection + scaled_dot_product_attention.

    Important:
        For maximum speed and clean profiling, keep need_attn_score=False.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        init_alpha: float = 1.0,
        alpha_eps: float = 1e-6,
        true_head_ratio: float = 0.5,
        need_attn_score: bool = False,
    ) -> None:
        super().__init__()

        assert embed_dim % num_heads == 0, (
            f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
        )
        assert init_alpha > 0, "init_alpha must be positive."
        assert 0.0 <= true_head_ratio <= 1.0, (
            f"true_head_ratio must be in [0, 1], got {true_head_ratio}"
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = float(dropout)
        self.alpha_eps = float(alpha_eps)
        self.true_head_ratio = float(true_head_ratio)
        self.need_attn_score = bool(need_attn_score)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        target = max(init_alpha - alpha_eps, 1e-6)
        self._init_log_alpha_value = math.log(math.expm1(target))

        self.log_alpha = nn.Parameter(
            torch.full(
                size=(embed_dim,),
                fill_value=self._init_log_alpha_value,
                dtype=torch.float32,
            )
        )

        true_head_num = int(round(num_heads * true_head_ratio))
        true_head_num = max(0, min(num_heads, true_head_num))

        true_head_mask = torch.zeros(num_heads, dtype=torch.bool)
        if true_head_num > 0:
            true_head_mask[:true_head_num] = True

        self.register_buffer("true_head_mask", true_head_mask)

        true_channel_mask = true_head_mask.repeat_interleave(self.head_dim)
        self.register_buffer("true_channel_mask", true_channel_mask)

        self.reset_parameters()

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.log_alpha) + self.alpha_eps

    @property
    def true_head_num(self) -> int:
        return int(self.true_head_mask.sum().item())

    @property
    def proxy_head_num(self) -> int:
        return int((~self.true_head_mask).sum().item())

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
        if self.k_proj.bias is not None:
            nn.init.zeros_(self.k_proj.bias)
        if self.v_proj.bias is not None:
            nn.init.zeros_(self.v_proj.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

        with torch.no_grad():
            self.log_alpha.fill_(self._init_log_alpha_value)

        # Strictly freeze Wq/bq.
        for param in self.q_proj.parameters():
            param.requires_grad_(False)

        self._save_initial_q_state()

    def _save_initial_q_state(self) -> None:
        with torch.no_grad():
            q_weight = self.q_proj.weight.detach().clone()

            if "_initial_q_weight" in self._buffers:
                self._buffers["_initial_q_weight"] = q_weight
            else:
                self.register_buffer("_initial_q_weight", q_weight)

            if self.q_proj.bias is not None:
                q_bias = self.q_proj.bias.detach().clone()

                if "_initial_q_bias" in self._buffers:
                    self._buffers["_initial_q_bias"] = q_bias
                else:
                    self.register_buffer("_initial_q_bias", q_bias)
            else:
                if "_initial_q_bias" not in self._buffers:
                    self.register_buffer("_initial_q_bias", None)

    @torch.no_grad()
    def restore_frozen_q(self) -> None:
        self.q_proj.weight.copy_(
            self._initial_q_weight.to(self.q_proj.weight.device)
        )

        if self.q_proj.bias is not None and self._initial_q_bias is not None:
            self.q_proj.bias.copy_(
                self._initial_q_bias.to(self.q_proj.bias.device)
            )

    @torch.no_grad()
    def check_q_change(self) -> tuple[float, float]:
        weight_diff_max = (
            self.q_proj.weight
            - self._initial_q_weight.to(self.q_proj.weight.device)
        ).abs().max().item()

        if self.q_proj.bias is not None and self._initial_q_bias is not None:
            bias_diff_max = (
                self.q_proj.bias
                - self._initial_q_bias.to(self.q_proj.bias.device)
            ).abs().max().item()
        else:
            bias_diff_max = 0.0

        return weight_diff_max, bias_diff_max

    def _shape_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        return (
            x.view(B, N, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, N, Dh = x.shape
        return (
            x.transpose(1, 2)
            .contiguous()
            .view(B, N, H * Dh)
        )

    def _build_additive_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        x: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Convert attention mask to additive format.

        Supported input:
            None
            [N, N]
            [B, N, N]
            [B, H, N, N]

        Mask convention:
            additive mask: 0 allowed, negative value blocked
            binary mask:   1 allowed, 0 blocked
        """
        if attention_mask is None:
            return None

        mask = attention_mask.to(device=x.device)

        # Additive mask: 0 / negative large value.
        if torch.is_floating_point(mask) and torch.min(mask) < 0:
            if mask.dim() == 2:
                N, M = mask.shape
                return mask.to(dtype=x.dtype).view(1, 1, N, M)

            if mask.dim() == 3:
                return mask.unsqueeze(1).to(dtype=x.dtype)

            if mask.dim() == 4:
                return mask.to(dtype=x.dtype)

            raise ValueError(
                f"Unsupported additive attention_mask shape: {attention_mask.shape}. "
                "Expected [N, N], [B, N, N], or [B, H, N, N]."
            )

        # Binary mask: 1 allowed, 0 blocked.
        if mask.dim() == 2:
            N = mask.shape[0]
            eye = torch.eye(N, device=mask.device, dtype=mask.dtype)
            mask = torch.maximum(mask, eye)
            additive_mask = (1.0 - mask.float()) * -1e9
            return additive_mask.to(dtype=x.dtype).view(1, 1, N, N)

        if mask.dim() == 3:
            B, N, _ = mask.shape
            eye = torch.eye(N, device=mask.device, dtype=mask.dtype).unsqueeze(0)
            mask = torch.maximum(mask, eye)
            additive_mask = (1.0 - mask.float()) * -1e9
            return additive_mask.unsqueeze(1).to(dtype=x.dtype)

        if mask.dim() == 4:
            return mask.to(dtype=x.dtype)

        raise ValueError(
            f"Unsupported attention_mask shape: {attention_mask.shape}. "
            "Expected [N, N], [B, N, N], or [B, H, N, N]."
        )

    def _manual_attention_with_score(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        additive_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if additive_mask is not None:
            score = score + additive_mask

        attn_score = torch.softmax(score, dim=-1)

        if self.training and self.dropout > 0.0:
            attn_score = F.dropout(
                attn_score,
                p=self.dropout,
                training=True,
            )

        out = torch.matmul(attn_score, v)

        return out, attn_score

    def _slow_attention_with_score(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Slow path only used when need_attn_score=True.

        Not intended for speed profiling.
        """
        alpha_channel = self.alpha.to(device=x.device, dtype=x.dtype)
        additive_mask = self._build_additive_attention_mask(attention_mask, x)

        x_qk = x.detach()

        with torch.no_grad():
            q = self.q_proj(x_qk)

        # This path is only for returning attention scores.
        # For speed experiments, use need_attn_score=False.
        k = self.k_proj(x_qk) * alpha_channel.detach().view(1, 1, -1)
        v = self.v_proj(x)

        q = self._shape_to_heads(q)
        k = self._shape_to_heads(k)
        v = self._shape_to_heads(v)

        out, attn_score = self._manual_attention_with_score(
            q=q,
            k=k,
            v=v,
            additive_mask=additive_mask,
        )

        out = self._merge_heads(out)
        out = self.out_proj(out)

        return out, attn_score

    def _fast_eval_attention(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        alpha_channel = self.alpha.to(device=x.device, dtype=x.dtype)
        additive_mask = self._build_additive_attention_mask(attention_mask, x)

        x_qk = x.detach()

        q = self.q_proj(x_qk)
        k = self.k_proj(x_qk) * alpha_channel.view(1, 1, -1)
        v = self.v_proj(x)

        q = self._shape_to_heads(q)
        k = self._shape_to_heads(k)
        v = self._shape_to_heads(v)

        if self.need_attn_score:
            out, attn_score = self._manual_attention_with_score(
                q=q,
                k=k,
                v=v,
                additive_mask=additive_mask,
            )
        else:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=additive_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            attn_score = None

        out = self._merge_heads(out)
        out = self.out_proj(out)

        return out, attn_score

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # Returning attention score is memory-heavy.
        # Use slow path only when explicitly requested.
        if self.need_attn_score:
            return self._slow_attention_with_score(
                x=x,
                attention_mask=attention_mask,
            )

        if not self.training:
            return self._fast_eval_attention(
                x=x,
                attention_mask=attention_mask,
            )

        additive_mask = self._build_additive_attention_mask(attention_mask, x)

        out = _ProxyKeyAttentionFunction.apply(
            x,
            self.q_proj.weight,
            self.q_proj.bias,
            self.k_proj.weight,
            self.k_proj.bias,
            self.v_proj.weight,
            self.v_proj.bias,
            self.log_alpha,
            additive_mask,
            self.num_heads,
            self.true_head_num,
            self.dropout,
            self.training,
            self.alpha_eps,
        )

        out = self.out_proj(out)

        return out, None


class Encoder_No_Grad_Test(nn.Module):
    """
    Encoder block with optimized custom-backward head-wise true-gradient / proxy-gradient attention.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int,
        dropout: float,
        init_alpha: float = 1.0,
        true_head_ratio: float = 0.5,
        need_attn_score: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        assert input_dim % num_heads == 0, (
            f"input_dim={input_dim} must be divisible by num_heads={num_heads}"
        )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.need_attn_score = bool(need_attn_score)
        self.true_head_ratio = float(true_head_ratio)

        self.attention = HeadwiseTrueGradientProxyKeyAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=True,
            init_alpha=init_alpha,
            true_head_ratio=true_head_ratio,
            need_attn_score=need_attn_score,
        )

        self.normalize_layer1 = nn.LayerNorm(normalized_shape=input_dim)
        self.normalize_layer2 = nn.LayerNorm(normalized_shape=input_dim)

        self.ffn = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim, out_features=output_dim),
            nn.Dropout(dropout),
        )

        self.residual_proj = (
            nn.Identity()
            if output_dim == input_dim
            else nn.Linear(input_dim, output_dim)
        )

        self._init_weight()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        normed_x = self.normalize_layer1(x)

        attn_out, attn_score = self.attention(
            x=normed_x,
            attention_mask=attention_mask,
        )

        attn_out = attn_out + x

        normed_attn_out = self.normalize_layer2(attn_out)
        ffn_out = self.ffn(normed_attn_out)

        x = ffn_out + self.residual_proj(attn_out)

        return x, attn_score

    def restore_frozen_q(self) -> None:
        self.attention.restore_frozen_q()

    def check_q_change(self) -> tuple[float, float]:
        return self.attention.check_q_change()

    def get_alpha(self) -> list[float]:
        return self.attention.alpha.detach().cpu().tolist()

    def get_true_head_ratio(self) -> float:
        return self.true_head_ratio

    def get_true_head_num(self) -> int:
        return self.attention.true_head_num

    def get_proxy_head_num(self) -> int:
        return self.attention.proxy_head_num

    def _init_weight(self) -> None:
        self.attention.reset_parameters()

        ffn_linear_index = [
            idx for idx, sub_layer in enumerate(self.ffn)
            if isinstance(sub_layer, nn.Linear)
        ]

        for idx, sub_layer in enumerate(self.ffn):
            if isinstance(sub_layer, nn.Linear):
                if idx != ffn_linear_index[-1]:
                    nn.init.kaiming_uniform_(
                        sub_layer.weight,
                        nonlinearity="relu",
                    )
                else:
                    nn.init.kaiming_uniform_(
                        sub_layer.weight,
                        mode="fan_in",
                    )

                if sub_layer.bias is not None:
                    nn.init.zeros_(sub_layer.bias)

        if isinstance(self.residual_proj, nn.Linear):
            nn.init.xavier_uniform_(self.residual_proj.weight)
            if self.residual_proj.bias is not None:
                nn.init.zeros_(self.residual_proj.bias)

        nn.init.constant_(self.normalize_layer1.weight, 1.0)
        nn.init.constant_(self.normalize_layer1.bias, 0.0)
        nn.init.constant_(self.normalize_layer2.weight, 1.0)
        nn.init.constant_(self.normalize_layer2.bias, 0.0)