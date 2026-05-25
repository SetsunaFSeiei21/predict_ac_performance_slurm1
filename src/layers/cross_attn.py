import torch
import torch.nn as nn
from typing import Tuple


class Cross_Attention(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        num_heads: int,
        **kwargs,
    ):
        super().__init__()

        assert input_dim % num_heads == 0, (
            f"input_dim={input_dim} must be divisible by num_heads={num_heads}"
        )

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.normalize_layer1 = nn.LayerNorm(normalized_shape=input_dim)

        self.multi_head_attention_layer = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

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
        structure_query: torch.Tensor,
        parameter_kv: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        structure_query: [B, N_query, D]
        parameter_kv:    [B, N_kv, D]
        attn_mask:       [N_query, N_kv] or compatible MultiheadAttention mask
        """

        normed_structure_query = self.normalize_layer1(structure_query)

        multi_attn_out, attn_score = self.multi_head_attention_layer(
            query=normed_structure_query,
            key=parameter_kv,
            value=parameter_kv,
            attn_mask=attn_mask,
        )

        multi_attn_out = multi_attn_out + structure_query

        normed_multi_attn_out = self.normalize_layer2(multi_attn_out)

        ffn_output = self.ffn(normed_multi_attn_out)

        final_result = ffn_output + self.residual_proj(multi_attn_out)
        
        return final_result, attn_score
        
    def _init_weight(self) -> None:
        # 初始化 FFN
        ffn_linear_idx = [
            idx for idx, sub_layer in enumerate(self.ffn)
            if isinstance(sub_layer, nn.Linear)
        ]

        for idx, sub_layer in enumerate(self.ffn):
            if isinstance(sub_layer, nn.Linear):
                if idx != ffn_linear_idx[-1]:
                    nn.init.kaiming_uniform_(sub_layer.weight, nonlinearity="relu")
                else:
                    nn.init.kaiming_uniform_(sub_layer.weight, mode="fan_in")

                if sub_layer.bias is not None:
                    nn.init.zeros_(sub_layer.bias)
        
        # 初始化 MultiheadAttention
        if hasattr(self.multi_head_attention_layer, "in_proj_weight"):
            nn.init.xavier_uniform_(self.multi_head_attention_layer.in_proj_weight)

            if self.multi_head_attention_layer.in_proj_bias is not None:
                nn.init.zeros_(self.multi_head_attention_layer.in_proj_bias)

        if hasattr(self.multi_head_attention_layer, "out_proj"):
            nn.init.xavier_uniform_(self.multi_head_attention_layer.out_proj.weight)

            if self.multi_head_attention_layer.out_proj.bias is not None:
                nn.init.zeros_(self.multi_head_attention_layer.out_proj.bias)

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