import torch
import torch.nn as nn
from typing import Tuple

class Encoder(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_heads: int, dropout: float, **kwargs) -> None:
        super().__init__()
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.normalize_layer1 = nn.BatchNorm1d(num_features=input_dim)
        self.normalize_layer2 = nn.BatchNorm1d(num_features=input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features= hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim, out_features=output_dim),
            nn.Dropout(dropout)
        )
        self._init_weight()
        
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # 前置BatchNorm
        normed_x = x.permute(0,2,1)
        normed_x = self.normalize_layer1(normed_x)
        normed_x = normed_x.permute(0,2,1)
        attn_out, attn_score = self.multi_head_attention(query = normed_x, key = normed_x, value = normed_x, attn_mask = attention_mask)
        attn_out = attn_out + x
        x = attn_out.permute(0,2,1)
        x = self.normalize_layer2(x)
        x = x.permute(0,2,1)
        ffn_out = self.ffn(x)
        x = ffn_out + attn_out
        
        return x, attn_score
        
    def _init_weight(self) -> None:
        # 初始化ffn
        ffn_linear_index = [idx for idx, sub_layer in enumerate(self.ffn) if isinstance(sub_layer, nn.Linear)]
        for idx, sub_layer in enumerate(self.ffn):
            if isinstance(sub_layer, nn.Linear):
                if idx != ffn_linear_index[-1]:
                    nn.init.kaiming_uniform_(sub_layer.weight, nonlinearity="relu")
                else:
                    nn.init.kaiming_uniform_(sub_layer.weight, mode="fan_in")
                if sub_layer.bias is not None:
                    nn.init.zeros_(sub_layer.bias)
        # 初始化 MultiheadAttention Network
        if hasattr(self.multi_head_attention, 'in_proj_weight'):
            nn.init.xavier_uniform_(self.multi_head_attention.in_proj_weight)
            if self.multi_head_attention.in_proj_bias is not None:
                nn.init.zeros_(self.multi_head_attention.in_proj_bias)
        if hasattr(self.multi_head_attention, 'out_proj'):
            nn.init.xavier_uniform_(self.multi_head_attention.out_proj.weight)
            if self.multi_head_attention.out_proj.bias is not None:
                nn.init.zeros_(self.multi_head_attention.out_proj.bias)
        
        # 初始化BatchNorm
        nn.init.constant_(self.normalize_layer1.weight, 1.0)
        nn.init.constant_(self.normalize_layer1.bias, 0.0)
        nn.init.constant_(self.normalize_layer2.weight, 1.0)
        nn.init.constant_(self.normalize_layer2.bias, 0.0)