import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankDiagonalProxyKeyAttention(nn.Module):
    """
    Low-rank diagonal proxy-gradient attention.

    Goal:
        Completely remove real Wk score-gradient.

    Forward:
        Q = sg(H) Wq^0
        K = sg(H) Wk diag(alpha)
        V = H Wv
        O = softmax(QK^T / sqrt(d) + mask) V Wo

    Backward approximation:
        Wq: frozen
        H : only receives gradient from V/content path
        Wk: receives proxy-gradient from Wv only

    Proxy rule:
        Let A = diag(alpha) + U V^T.

        We construct:
            Wv_eff = Wv + A Wk - stopgrad(A Wk)

        Forward:
            Wv_eff == Wv

        Backward:
            grad(Wk) = A^T grad(Wv_eff)

        Therefore:
            grad(Wk) ≈ [diag(alpha) + U V^T]^T grad(Wv)

    Compared with diagonal proxy:
        diagonal proxy:
            grad(Wk) ≈ diag(alpha) grad(Wv)

        low-rank diagonal proxy:
            grad(Wk) ≈ [diag(alpha) + V U^T] grad(Wv)

    This gives the proxy-gradient channel-mixing ability without computing
    the real Wk score-gradient.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        init_alpha: float = 1.0,
        alpha_eps: float = 1e-6,
        proxy_rank: int = 16,
        low_rank_scale: float = 1.0,
        low_rank_init_std: float = 1e-3,
        need_attn_score: bool = False,
    ) -> None:
        super().__init__()

        assert embed_dim % num_heads == 0, (
            f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
        )
        assert init_alpha > 0, "init_alpha must be positive."
        assert proxy_rank > 0, "proxy_rank must be positive."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.alpha_eps = alpha_eps
        self.proxy_rank = proxy_rank
        self.low_rank_scale = low_rank_scale
        self.low_rank_init_std = low_rank_init_std
        self.need_attn_score = need_attn_score

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # alpha_c = softplus(log_alpha_c) + eps
        target = max(init_alpha - alpha_eps, 1e-6)
        self._init_log_alpha_value = math.log(math.expm1(target))

        self.log_alpha = nn.Parameter(
            torch.full(
                size=(embed_dim,),
                fill_value=self._init_log_alpha_value,
                dtype=torch.float32,
            )
        )

        # Low-rank proxy matrix:
        #   A = diag(alpha) + low_rank_scale * U V^T
        #
        # U is initialized to zero, V is initialized small random.
        # Therefore, at initialization:
        #   A ≈ diag(alpha)
        #
        # But U can receive gradient immediately because V is non-zero.
        self.proxy_u = nn.Parameter(torch.zeros(embed_dim, proxy_rank))
        self.proxy_v = nn.Parameter(
            torch.empty(embed_dim, proxy_rank)
        )

        self.reset_parameters()

    @property
    def alpha(self) -> torch.Tensor:
        """
        Shape:
            [embed_dim]
        """
        return F.softplus(self.log_alpha) + self.alpha_eps

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
            self.proxy_u.zero_()
            nn.init.normal_(
                self.proxy_v,
                mean=0.0,
                std=self.low_rank_init_std,
            )

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

    def _apply_proxy_matrix_to_weight(
        self,
        weight: torch.Tensor,
        alpha_channel: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply A Wk without explicitly constructing full A.

        A = diag(alpha) + low_rank_scale * U V^T

        weight:
            [D, D]

        return:
            [D, D]
        """
        diag_part = alpha_channel.view(-1, 1) * weight

        # V^T Wk: [r, D]
        low_rank_mid = self.proxy_v.transpose(0, 1) @ weight

        # U (V^T Wk): [D, D]
        low_rank_part = self.proxy_u @ low_rank_mid

        return diag_part + self.low_rank_scale * low_rank_part

    def _apply_proxy_matrix_to_bias(
        self,
        bias: torch.Tensor,
        alpha_channel: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply A bk without explicitly constructing full A.

        bias:
            [D]

        return:
            [D]
        """
        diag_part = alpha_channel * bias

        # V^T b: [r]
        low_rank_mid = self.proxy_v.transpose(0, 1) @ bias

        # U (V^T b): [D]
        low_rank_part = self.proxy_u @ low_rank_mid

        return diag_part + self.low_rank_scale * low_rank_part

    def _shape_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B, N, D] -> [B, H, N, Dh]
        """
        B, N, D = x.shape
        return (
            x.view(B, N, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B, H, N, Dh] -> [B, N, D]
        """
        B, H, N, Dh = x.shape
        return (
            x.transpose(1, 2)
            .contiguous()
            .view(B, N, H * Dh)
        )

    def _build_sdpa_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        x: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Compatible with torch.nn.functional.scaled_dot_product_attention.

        Supported input:
            None
            [N, N]
            [B, N, N]

        Mask convention:
            additive mask: 0 allowed, negative value blocked
            binary mask:   1 allowed, 0 blocked
        """
        if attention_mask is None:
            return None

        mask = attention_mask.to(device=x.device)

        # Additive mask: 0 / -1e9
        if torch.is_floating_point(mask) and torch.min(mask) < 0:
            if mask.dim() == 2:
                return mask.to(dtype=x.dtype)

            if mask.dim() == 3:
                return mask.unsqueeze(1).to(dtype=x.dtype)

            raise ValueError(
                f"Unsupported additive attention_mask shape: {attention_mask.shape}. "
                "Expected [N, N] or [B, N, N]."
            )

        # Binary mask: 1 allowed, 0 blocked.
        if mask.dim() == 2:
            N = mask.shape[0]
            eye = torch.eye(N, device=mask.device, dtype=mask.dtype)
            mask = torch.maximum(mask, eye)
            additive_mask = (1.0 - mask.float()) * -1e9
            return additive_mask.to(dtype=x.dtype)

        if mask.dim() == 3:
            B, N, _ = mask.shape
            eye = torch.eye(N, device=mask.device, dtype=mask.dtype).unsqueeze(0)
            mask = torch.maximum(mask, eye)
            additive_mask = (1.0 - mask.float()) * -1e9
            return additive_mask.unsqueeze(1).to(dtype=x.dtype)

        raise ValueError(
            f"Unsupported attention_mask shape: {attention_mask.shape}. "
            "Expected [N, N] or [B, N, N]."
        )

    def _manual_attention_with_score(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            score = score + attn_mask

        attn_score = torch.softmax(score, dim=-1)

        if self.training and self.dropout > 0:
            attn_score = F.dropout(attn_score, p=self.dropout, training=True)

        out = torch.matmul(attn_score, v)
        return out, attn_score

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x:
            [B, N, D]
        """
        alpha_channel = self.alpha.to(device=x.device, dtype=x.dtype)

        # Q/K use detached hidden states, so Q/K paths do not send gradients to H.
        x_qk = x.detach()

        # Q is frozen and no graph is needed.
        # K is also computed under no_grad, so Wk receives no real score-gradient.
        with torch.no_grad():
            q = self.q_proj(x_qk)
            k_raw = self.k_proj(x_qk)

        # K forward uses alpha only as a forward scaling.
        # Detach alpha here so alpha itself is not trained by score-gradient.
        k = k_raw * alpha_channel.detach().view(1, 1, -1)

        # Low-rank diagonal proxy-gradient path.
        #
        # Forward:
        #   proxy_weight - proxy_weight.detach() == 0
        #   v_weight_eff == Wv
        #
        # Backward:
        #   grad(Wk), grad(alpha), grad(U), grad(V)
        #   are obtained from the V/content path through proxy_weight.
        proxy_weight = self._apply_proxy_matrix_to_weight(
            self.k_proj.weight,
            alpha_channel,
        )

        v_weight_eff = self.v_proj.weight + proxy_weight - proxy_weight.detach()

        if self.v_proj.bias is not None:
            proxy_bias = self._apply_proxy_matrix_to_bias(
                self.k_proj.bias,
                alpha_channel,
            )
            v_bias_eff = self.v_proj.bias + proxy_bias - proxy_bias.detach()
        else:
            v_bias_eff = None

        v = F.linear(x, v_weight_eff, v_bias_eff)

        q = self._shape_to_heads(q)
        k = self._shape_to_heads(k)
        v = self._shape_to_heads(v)

        sdpa_mask = self._build_sdpa_mask(attention_mask, x)

        if self.need_attn_score:
            out, attn_score = self._manual_attention_with_score(
                q=q,
                k=k,
                v=v,
                attn_mask=sdpa_mask,
            )
        else:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=sdpa_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
            attn_score = None

        out = self._merge_heads(out)
        out = self.out_proj(out)

        return out, attn_score


class Encoder_No_Grad_Test(nn.Module):
    """
    Encoder block with low-rank diagonal proxy-gradient attention.

    Main approximation:
        grad(Wk) ≈ [diag(alpha) + U V^T]^T grad(Wv)

    Removed:
        Q/K score-gradient-to-hidden path
        Wq update
        real Wk score-gradient

    Preserved:
        forward topology-aware attention routing
        V/content-gradient-to-hidden path
        proxy Wk update from Wv
        out projection / FFN / LayerNorm
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int,
        dropout: float,
        init_alpha: float = 1.0,
        proxy_rank: int = 16,
        low_rank_scale: float = 1.0,
        low_rank_init_std: float = 1e-3,
        need_attn_score: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        assert input_dim % num_heads == 0, (
            f"input_dim={input_dim} must be divisible by num_heads={num_heads}"
        )

        assert output_dim == input_dim, (
            f"output_dim={output_dim} must equal input_dim={input_dim} "
            "because residual connection uses ffn_out + attn_out."
        )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.proxy_rank = proxy_rank
        self.need_attn_score = need_attn_score

        self.attention = LowRankDiagonalProxyKeyAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=True,
            init_alpha=init_alpha,
            proxy_rank=proxy_rank,
            low_rank_scale=low_rank_scale,
            low_rank_init_std=low_rank_init_std,
            need_attn_score=need_attn_score,
        )

        self.normalize_layer1 = nn.LayerNorm(input_dim)
        self.normalize_layer2 = nn.LayerNorm(input_dim)

        self.ffn = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim, out_features=output_dim),
            nn.Dropout(dropout),
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

        x = ffn_out + attn_out

        return x, attn_score

    def restore_frozen_q(self) -> None:
        self.attention.restore_frozen_q()

    def check_q_change(self) -> tuple[float, float]:
        return self.attention.check_q_change()

    def get_alpha(self) -> list[float]:
        return self.attention.alpha.detach().cpu().tolist()

    def get_proxy_rank(self) -> int:
        return self.proxy_rank

    def _init_weight(self) -> None:
        self.attention.reset_parameters()

        ffn_linear_index = [
            idx for idx, sub_layer in enumerate(self.ffn)
            if isinstance(sub_layer, nn.Linear)
        ]

        for idx, sub_layer in enumerate(self.ffn):
            if isinstance(sub_layer, nn.Linear):
                if idx != ffn_linear_index[-1]:
                    nn.init.kaiming_uniform_(sub_layer.weight, nonlinearity="relu")
                else:
                    nn.init.kaiming_uniform_(sub_layer.weight, mode="fan_in")

                if sub_layer.bias is not None:
                    nn.init.zeros_(sub_layer.bias)

        nn.init.constant_(self.normalize_layer1.weight, 1.0)
        nn.init.constant_(self.normalize_layer1.bias, 0.0)
        nn.init.constant_(self.normalize_layer2.weight, 1.0)
        nn.init.constant_(self.normalize_layer2.bias, 0.0)