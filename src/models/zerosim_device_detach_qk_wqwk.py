import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any

from src.base import Device_BaseModel
from src.utils import get_mlp_layer
from src.layers import Decoder
from src.modules import Structure_Encoding_Layer_Detach_qk_wqwk, Parameter_Injection_Layer_Detach_qk_wqwk

class Zerosim_Device_Detach_qk_wqwk(Device_BaseModel):
    
    def __init__(self, feature_dim: int, hidden_dim: int, output_dim: int, dropout: float, num_heads: int, embedding_layer_num: int, structure_encoding_layer_num: int, 
                parameter_injection_layer_num: int, decoder_layer_num: int, output_layer_num: int, performance_num: int, 
                device_messages: List[Dict[str, Any]], adj_mask: np.ndarray) -> None:
        
        super().__init__(device_messages)
        assert hidden_dim % num_heads == 0, (f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.device_level_one_hot_tensors: torch.Tensor
        self.global_token = nn.Parameter(torch.empty(1, hidden_dim))
        self.performance_metric = nn.Parameter(torch.empty(performance_num, hidden_dim))
        attn_mask = self._get_attn_mask(torch.as_tensor(adj_mask, dtype=torch.float32).clone())
        pr_attn_mask = self._get_pr_attn_mask(adj_mask.shape[0])
        self.register_buffer("attn_mask", attn_mask)
        self.register_buffer("pr_attn_mask", pr_attn_mask)
        self.network = nn.ModuleDict({
            "device_embedding_layer": get_mlp_layer(input_dim=self.device_level_one_hot_tensors.shape[1], hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout,
                                                    layer_num=embedding_layer_num),
            "parameter_embedding_layer": get_mlp_layer(input_dim=feature_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout, layer_num=embedding_layer_num),
            "structure_encoding_layer": nn.ModuleList([
                Structure_Encoding_Layer_Detach_qk_wqwk(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout, num_heads=num_heads, attn_mask=attn_mask)
                for _ in range(structure_encoding_layer_num)
            ]),
            "parameter_injection_layer": nn.ModuleList([
                Parameter_Injection_Layer_Detach_qk_wqwk(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout, num_heads=num_heads, attn_mask=self.attn_mask,
                                        pr_attn_mask=self.pr_attn_mask) for _ in range(parameter_injection_layer_num)
            ]),
            "decoder": nn.ModuleList([
                Decoder(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout, num_heads=num_heads) for _ in range(decoder_layer_num)
            ]),
            "output_layer": get_mlp_layer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout, layer_num=output_layer_num)
        })
        self._init_weight()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        B, _, _ = x.shape
        device_tensors = self.network["device_embedding_layer"](self.device_level_one_hot_tensors).unsqueeze(0).expand(B, -1, -1)
        global_tensors = self.global_token.unsqueeze(0).expand(B, -1, -1)
        device_tensors = torch.cat([device_tensors, global_tensors], dim=1)
        parameter_tensors = self.network["parameter_embedding_layer"](x)
        for sub_layer in self.network["structure_encoding_layer"]:
            device_tensors = sub_layer(device_tensors)
        for sub_layer in self.network["parameter_injection_layer"]:
            device_tensors = sub_layer(device_tensors, parameter_tensors)
        performance_tensors = self.performance_metric.unsqueeze(0).expand(B, -1, -1)
        for sub_layer in self.network["decoder"]:
            performance_tensors, _ = sub_layer(device_tensors, performance_tensors)
        output_ternsors = self.network['output_layer'](performance_tensors).reshape(B, -1)
        
        return output_ternsors
        
    def _get_pr_attn_mask(self, device_num: int) -> torch.Tensor:
        
        initial_mask = torch.eye(device_num, dtype=torch.float32)
        pr_attn_mask = torch.cat((initial_mask, torch.ones(1, device_num)), dim=0)
        
        return (pr_attn_mask - 1) * 1e9
        
    def _get_attn_mask(self, adj: torch.Tensor) -> torch.Tensor:
        
        N, _ = adj.shape
        add_row_tensors = torch.ones((1, N))
        add_col_tensors = torch.ones((N+1, 1))
        adj_tensors = torch.cat((adj, add_row_tensors), dim=0)
        adj_tensors = torch.cat((adj_tensors, add_col_tensors), dim=-1)
        
        return (adj_tensors - 1) * 1e9
        
    def _init_weight(self,) -> None:
        
        nn.init.normal_(self.global_token, mean=0.0, std=0.02)
        nn.init.normal_(self.performance_metric, mean=0.0, std=0.02)