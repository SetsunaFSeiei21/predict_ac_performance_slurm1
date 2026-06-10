import torch
import torch.nn as nn
from typing import Tuple


class Encoder_Detach_qk(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int,
        dropout: float,
        **kwargs
    ) -> None:
        super().__init__()

        assert input_dim % num_heads == 0, (
            f"input_dim={input_dim} must be divisible by num_heads={num_heads}"
        )

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # LayerNorm 直接作用在最后一维，不需要 permute
        self.normalize_layer1 = nn.LayerNorm(normalized_shape=input_dim)
        self.normalize_layer2 = nn.LayerNorm(normalized_shape=input_dim)

        self.ffn = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim, out_features=output_dim),
            nn.Dropout(dropout)
        )

        # 如果 output_dim != input_dim，FFN 残差需要投影
        self.residual_proj = (
            nn.Identity()
            if output_dim == input_dim
            else nn.Linear(input_dim, output_dim)
        )

        self._init_weight()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # x: [batch_size, node_num, input_dim]

        # Pre-LayerNorm + Multi-Head Attention
        normed_x = self.normalize_layer1(x)

        attn_out, attn_score = self.multi_head_attention(
            query=normed_x.detach(),
            key=normed_x.detach(),
            value=normed_x,
            attn_mask=attention_mask
        )

        # Attention residual
        attn_out = attn_out + x

        # Pre-LayerNorm + FFN
        normed_attn_out = self.normalize_layer2(attn_out)
        ffn_out = self.ffn(normed_attn_out)

        # FFN residual
        x = ffn_out + self.residual_proj(attn_out)

        return x, attn_score

    def _init_weight(self) -> None:
        # 初始化 FFN
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

        # 初始化 MultiheadAttention
        if hasattr(self.multi_head_attention, "in_proj_weight"):
            nn.init.xavier_uniform_(self.multi_head_attention.in_proj_weight)

            if self.multi_head_attention.in_proj_bias is not None:
                nn.init.zeros_(self.multi_head_attention.in_proj_bias)

        if hasattr(self.multi_head_attention, "out_proj"):
            nn.init.xavier_uniform_(self.multi_head_attention.out_proj.weight)

            if self.multi_head_attention.out_proj.bias is not None:
                nn.init.zeros_(self.multi_head_attention.out_proj.bias)

        # 初始化 residual projection
        if isinstance(self.residual_proj, nn.Linear):
            nn.init.xavier_uniform_(self.residual_proj.weight)
            if self.residual_proj.bias is not None:
                nn.init.zeros_(self.residual_proj.bias)

        # 初始化 LayerNorm
        nn.init.constant_(self.normalize_layer1.weight, 1.0)
        nn.init.constant_(self.normalize_layer1.bias, 0.0)
        nn.init.constant_(self.normalize_layer2.weight, 1.0)
        nn.init.constant_(self.normalize_layer2.bias, 0.0)