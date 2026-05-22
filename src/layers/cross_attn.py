import torch
import torch.nn as nn
from typing import Tuple

class Cross_Attention(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float, num_heads: int, **kwargs):
        
        super().__init__()
        self.normalize_layer1 = nn.BatchNorm1d(num_features=input_dim)
        self.multi_head_attention_layer = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.normalize_layer2 = nn.BatchNorm1d(num_features=input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim, out_features=output_dim),
            nn.Dropout(dropout)
        )
        self._init_weight()
        
    def forward(self, structure_query: torch.Tensor, parameter_kv: torch.Tensor,
                attn_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        normed_structure_query = structure_query.permute(0, 2, 1)
        normed_structure_query = self.normalize_layer1(normed_structure_query)
        normed_structure_query = normed_structure_query.permute(0, 2, 1)
        multi_attn_out, attn_score = self.multi_head_attention_layer(query = normed_structure_query, key = parameter_kv, value = parameter_kv, attn_mask = attn_mask)
        multi_attn_out = multi_attn_out + structure_query
        normed_multi_attn_out = multi_attn_out.permute(0, 2, 1)
        normed_multi_attn_out = self.normalize_layer2(normed_multi_attn_out)
        normed_multi_attn_out = normed_multi_attn_out.permute(0, 2, 1)
        ffn_output = self.ffn(normed_multi_attn_out)
        final_result = ffn_output + multi_attn_out
        
        return final_result, attn_score
        
    def _init_weight(self) -> None:

        ffn_linear_idx = [idx for idx, sub_layer in enumerate(self.ffn) if isinstance(sub_layer, nn.Linear)]
        for idx, sub_layer in enumerate(self.ffn):
            if isinstance(sub_layer, nn.Linear):
                if idx != ffn_linear_idx[-1]:
                    nn.init.kaiming_uniform_(sub_layer.weight, nonlinearity='relu')
                else:
                    nn.init.kaiming_uniform_(sub_layer.weight, mode='fan_in')
                if sub_layer.bias is not None:
                    nn.init.zeros_(sub_layer.bias)
        
        # 初始化Multi_head_Attention Layer
        if hasattr(self.multi_head_attention_layer, 'in_proj_weight'):
            nn.init.xavier_uniform_(self.multi_head_attention_layer.in_proj_weight)
            if hasattr(self.multi_head_attention_layer, 'in_proj_bias'):
                nn.init.zeros_(self.multi_head_attention_layer.in_proj_bias)
        if hasattr(self.multi_head_attention_layer, 'out_proj'):
            nn.init.xavier_uniform_(self.multi_head_attention_layer.out_proj.weight)
            if self.multi_head_attention_layer.out_proj.bias is not None:
                nn.init.zeros_(self.multi_head_attention_layer.out_proj.bias)
                
        # 初始化BatchNorm
        nn.init.constant_(self.normalize_layer1.weight, 1.0)
        nn.init.constant_(self.normalize_layer1.bias, 0.0)
        nn.init.constant_(self.normalize_layer2.weight, 1.0)
        nn.init.constant_(self.normalize_layer2.bias, 0.0)