import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any

from src.layers import Decoder
from src.base import Device_BaseModel
from src.utils import get_mlp_layer

class Ablation_GE_WO_CG(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_heads: int, dropout: float, **kwargs):
        
        assert output_dim == input_dim, (f"output_dim={output_dim} should squal to input_dim={input_dim}")
        super().__init__()
        self.multi_head_attention_layer = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.normalize_layer1 = nn.BatchNorm1d(num_features=input_dim)
        self.normalize_layer2 = nn.BatchNorm1d(num_features=input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim, out_features=output_dim),
            nn.Dropout(dropout)
        )
        self._init_weight()
        
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.normalize_layer1(x)
        x = x.permute(0, 2, 1)
        attn_out, attn_score = self.multi_head_attention_layer(query = x, key = x, value = x, attn_mask = attention_mask)
        x = x + attn_out
        x = x.permute(0, 2, 1)
        x = self.normalize_layer2(x)
        x = x.permute(0, 2, 1)
        ffn_out = self.ffn(x)
        x = ffn_out + x

        return x, attn_score
                
    def _init_weight(self) -> None:
        # ffn初始化
        ffn_linear_layer_index_lst = [idx for idx, sub_layer in enumerate(self.ffn) if isinstance(sub_layer, nn.Linear)]
        for idx, sub_layer in enumerate(self.ffn):
            if isinstance(sub_layer, nn.Linear):
                if idx != ffn_linear_layer_index_lst[-1]:
                    nn.init.kaiming_uniform_(sub_layer.weight, nonlinearity='relu')
                else:
                    nn.init.kaiming_uniform_(sub_layer.weight, mode='fan_in')
                if sub_layer.bias is not None:
                    nn.init.zeros_(sub_layer.bias)
        
        # 多头注意力层初始化
        if hasattr(self.multi_head_attention_layer, 'in_proj_weight'):
            nn.init.xavier_uniform_(self.multi_head_attention_layer.in_proj_weight)
            if self.multi_head_attention_layer.in_proj_bias is not None:
                nn.init.zeros_(self.multi_head_attention_layer.in_proj_bias)
        if hasattr(self.multi_head_attention_layer, 'out_proj'):
            nn.init.xavier_uniform_(self.multi_head_attention_layer.out_proj.weight)
            if self.multi_head_attention_layer.out_proj.bias is not None:
                nn.init.zeros_(self.multi_head_attention_layer.out_proj.bias)
                
        # normalize_layer初始化
        nn.init.constant_(self.normalize_layer1.weight, 1.0)
        nn.init.constant_(self.normalize_layer1.bias, 0.0)
        nn.init.constant_(self.normalize_layer2.weight, 1.0)
        nn.init.constant_(self.normalize_layer2.bias, 0.0)

class Ablation_Global_Encoder_WO_CG_WO_SE(Device_BaseModel):
    
    def __init__(self, feature_dim: int, hidden_dim: int, output_dim: int, dropout: float, num_heads: int, 
                embedding_layer_num: int, encoder_layer_num: int, unmask_encoder_layer_num: int, decoder_layer_num: int, output_layer_num: int, 
                performance_num: int, device_messages: List[Dict[str, Any]], adj_mask: np.ndarray) -> None:
        
        assert hidden_dim % num_heads == 0, (f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        assert encoder_layer_num >= unmask_encoder_layer_num, (f"Encoder_layer_num={encoder_layer_num} should not smaller\
                                                                    than unmask_global_encoder_layer_num={unmask_encoder_layer_num}")
        super().__init__(device_messages)
        self.device_level_one_hot_tensors: torch.Tensor
        self.unmask_encoder_layer_num = unmask_encoder_layer_num
        attn_mask = self._get_attn_mask(torch.as_tensor(adj_mask, dtype=torch.float32).clone())
        self.register_buffer("attn_mask", attn_mask)
        self.global_token = nn.Parameter(torch.empty(1, hidden_dim))
        self.performance_metric = nn.Parameter(torch.empty(performance_num, hidden_dim))
        self.network = nn.ModuleDict({
            "device_embedding_layer": get_mlp_layer(input_dim=self.device_level_one_hot_tensors.shape[1], hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout,
                                                    layer_num=embedding_layer_num),
            "parameter_embedding_layer": get_mlp_layer(input_dim=feature_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout, layer_num=embedding_layer_num),
            "encoder_layer": nn.ModuleList([
                Ablation_GE_WO_CG(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, num_heads=num_heads, dropout=dropout) for _ in range(encoder_layer_num)
                ]), 
            "decoder_layer": nn.ModuleList([
                Decoder(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout, num_heads=num_heads) for _ in range(decoder_layer_num)
            ]),
            "output_layer": get_mlp_layer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout, layer_num=output_layer_num)
        })
        self._init_weight()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        B, _, _ = x.shape
        final_one_hot_device_tensors = self.device_level_one_hot_tensors.unsqueeze(0).expand(B, -1, -1)
        device_embeddings = self.network["device_embedding_layer"](final_one_hot_device_tensors)
        parameter_embeddings = self.network["parameter_embedding_layer"](x)
        shared_global_controller = self.global_token.reshape(1,1,-1).expand(B, -1, -1)
        masked_layer_num = len(self.network["encoder_layer"]) - self.unmask_encoder_layer_num
        for idx, sub_layer in enumerate(self.network["encoder_layer"]):
            encoder_input = torch.cat((device_embeddings, shared_global_controller),dim=1)
            if idx < masked_layer_num:
                encoder_output, _ = sub_layer(encoder_input, self.attn_mask)
            else:
                encoder_output, _ = sub_layer(encoder_input)
            device_embeddings = encoder_output[:, :-1, :] + parameter_embeddings
        performance_tensors = self.performance_metric.unsqueeze(0).expand(B, -1, -1)
        for sub_layer in self.network["decoder_layer"]:
            performance_tensors, _ = sub_layer(device_embeddings, performance_tensors)
        output_tensors = self.network["output_layer"](performance_tensors).reshape(B, -1)
        
        return output_tensors
    
    def _get_attn_mask(self, adj_mask: torch.Tensor) -> torch.Tensor:
        
        N, _ = adj_mask.shape
        col_tensors = torch.zeros((N, 1), dtype=torch.float32, device=adj_mask.device)
        row_tensors = torch.ones((1, N+1), dtype=torch.float32, device=adj_mask.device)
        tmp_attn_mask = torch.cat((adj_mask, col_tensors), dim=-1)
        attn_mask = torch.cat((tmp_attn_mask, row_tensors), dim=0)
        
        return (attn_mask - 1) * 1e9
        
    def _init_weight(self,) -> None:
        
        nn.init.normal_(self.global_token, mean=0.0, std=0.02)
        nn.init.normal_(self.performance_metric, mean = 0.0, std=0.02)