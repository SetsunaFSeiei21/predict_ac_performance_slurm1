import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAttention(nn.Module):
    """
    Linear attention baseline with ELU+1 feature map.

    Input:
        x: [B, N, D]

    Output:
        out: [B, N, D]

    Note:
        This is a forward-approximation attention baseline.
        It does not implement arbitrary topology-masked attention.
        The attn_mask argument is kept only for interface compatibility.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        assert hidden_dim % num_heads == 0, (
            f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
        )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.eps = eps

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self._init_weight()

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        # Positive feature map used by kernelized linear attention.
        return F.elu(x) + 1.0

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, N, D] -> [B, H, N, Dh]
        B, N, D = x.shape
        x = x.view(B, N, self.num_heads, self.head_dim)
        return x.transpose(1, 2).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, H, N, Dh] -> [B, N, D]
        B, H, N, Dh = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(B, N, H * Dh)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        """
        Args:
            x: [B, N, D]
            attn_mask: ignored. Kept for interface compatibility.

        Returns:
            out: [B, N, D]
            None: placeholder for attention score
        """
        q = self._split_heads(self.q_proj(x))  # [B, H, N, Dh]
        k = self._split_heads(self.k_proj(x))  # [B, H, N, Dh]
        v = self._split_heads(self.v_proj(x))  # [B, H, N, Dh]

        q = self._feature_map(q)
        k = self._feature_map(k)

        # [B, H, Dh, Dh]
        kv = torch.einsum("bhnd,bhne->bhde", k, v)

        # [B, H, Dh]
        k_sum = k.sum(dim=2)

        # [B, H, N]
        normalizer = 1.0 / (
            torch.einsum("bhnd,bhd->bhn", q, k_sum) + self.eps
        )

        # [B, H, N, Dh]
        out = torch.einsum("bhnd,bhde,bhn->bhne", q, kv, normalizer)

        out = self._merge_heads(out)
        out = self.dropout(out)
        out = self.out_proj(out)

        return out, None

    def _init_weight(self) -> None:
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


class LowRankAttention(nn.Module):
    """
    Linformer-style low-rank attention baseline.

    Input:
        x: [B, N, D]

    Output:
        out: [B, N, D]

    Note:
        num_nodes must be fixed.
        This is a forward-approximation attention baseline.
        It does not implement arbitrary topology-masked attention.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_nodes: int,
        proj_rank: int = 8,
        dropout: float = 0.0,
        share_kv_proj: bool = False,
    ) -> None:
        super().__init__()

        assert hidden_dim % num_heads == 0, (
            f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
        )
        assert proj_rank <= num_nodes, (
            f"proj_rank={proj_rank} must be <= num_nodes={num_nodes}"
        )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_nodes = num_nodes
        self.proj_rank = proj_rank
        self.head_dim = hidden_dim // num_heads
        self.share_kv_proj = share_kv_proj

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Project node dimension from N to R.
        # E: [H, R, N]
        self.E = nn.Parameter(torch.empty(num_heads, proj_rank, num_nodes))

        if share_kv_proj:
            self.F_proj = None
        else:
            self.F_proj = nn.Parameter(torch.empty(num_heads, proj_rank, num_nodes))

        self.dropout = nn.Dropout(dropout)

        self._init_weight()

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, N, D] -> [B, H, N, Dh]
        B, N, D = x.shape
        x = x.view(B, N, self.num_heads, self.head_dim)
        return x.transpose(1, 2).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, H, N, Dh] -> [B, N, D]
        B, H, N, Dh = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(B, N, H * Dh)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        """
        Args:
            x: [B, N, D]
            attn_mask: ignored. Kept for interface compatibility.

        Returns:
            out: [B, N, D]
            None: placeholder for attention score
        """
        B, N, D = x.shape

        if N != self.num_nodes:
            raise ValueError(
                f"LowRankAttention initialized with num_nodes={self.num_nodes}, "
                f"but got input with N={N}."
            )

        q = self._split_heads(self.q_proj(x))  # [B, H, N, Dh]
        k = self._split_heads(self.k_proj(x))  # [B, H, N, Dh]
        v = self._split_heads(self.v_proj(x))  # [B, H, N, Dh]

        # E:     [H, R, N]
        # k:     [B, H, N, Dh]
        # k_low: [B, H, R, Dh]
        k_low = torch.einsum("hrn,bhnd->bhrd", self.E, k)

        if self.share_kv_proj:
            v_low = torch.einsum("hrn,bhnd->bhrd", self.E, v)
        else:
            v_low = torch.einsum("hrn,bhnd->bhrd", self.F_proj, v)

        # scores: [B, H, N, R]
        scores = torch.einsum("bhnd,bhrd->bhnr", q, k_low)
        scores = scores / math.sqrt(self.head_dim)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # out: [B, H, N, Dh]
        out = torch.einsum("bhnr,bhrd->bhnd", attn, v_low)

        out = self._merge_heads(out)
        out = self.out_proj(out)

        return out, None

    def _init_weight(self) -> None:
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.E)
        if self.F_proj is not None:
            nn.init.xavier_uniform_(self.F_proj)


class EfficientEncoder(nn.Module):
    """
    Drop-in replacement for src.layers.Encoder.

    Same interface as Encoder:
        forward(x, attention_mask=None) -> (x, attn_score)

    attention_type:
        "linear"   -> LinearAttention
        "low_rank" -> LowRankAttention
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int,
        dropout: float,
        attention_type: str,
        num_nodes: Optional[int] = None,
        proj_rank: int = 8,
        **kwargs,
    ) -> None:
        super().__init__()

        assert input_dim % num_heads == 0, (
            f"input_dim={input_dim} must be divisible by num_heads={num_heads}"
        )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.attention_type = attention_type

        if attention_type == "linear":
            self.attention = LinearAttention(
                hidden_dim=input_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
        elif attention_type == "low_rank":
            if num_nodes is None:
                raise ValueError("num_nodes must be provided for LowRankAttention.")
            self.attention = LowRankAttention(
                hidden_dim=input_dim,
                num_heads=num_heads,
                num_nodes=num_nodes,
                proj_rank=proj_rank,
                dropout=dropout,
            )
        else:
            raise ValueError(
                f"Unknown attention_type={attention_type}. "
                "Expected 'linear' or 'low_rank'."
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
    ) -> Tuple[torch.Tensor, None]:
        # x: [B, N, D]

        normed_x = self.normalize_layer1(x)

        # attention_mask is ignored by LinearAttention / LowRankAttention.
        attn_out, attn_score = self.attention(
            normed_x,
            attn_mask=attention_mask,
        )

        # Attention residual
        attn_out = attn_out + x

        # FFN residual
        normed_attn_out = self.normalize_layer2(attn_out)
        ffn_out = self.ffn(normed_attn_out)
        out = ffn_out + self.residual_proj(attn_out)

        return out, attn_score

    def _init_weight(self) -> None:
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

        if isinstance(self.residual_proj, nn.Linear):
            nn.init.xavier_uniform_(self.residual_proj.weight)
            if self.residual_proj.bias is not None:
                nn.init.zeros_(self.residual_proj.bias)

        nn.init.constant_(self.normalize_layer1.weight, 1.0)
        nn.init.constant_(self.normalize_layer1.bias, 0.0)
        nn.init.constant_(self.normalize_layer2.weight, 1.0)
        nn.init.constant_(self.normalize_layer2.bias, 0.0)