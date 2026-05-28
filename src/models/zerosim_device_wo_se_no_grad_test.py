import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any

from src.base import Device_BaseModel
from src.utils import get_mlp_layer
from src.layers import Decoder
from src.modules import Parameter_Injection_Layer_No_Grad_Test

class Zerosim_Device_WO_SE_No_Grad_Test(Device_BaseModel):
    
    """_summary_
    这个模型为了探究zerosim在没有structure encoding部分的效果
    Args:
        Device_BaseModel (_type_): _description_
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int, output_dim: int, dropout: float, num_heads: int, embedding_layer_num: int, 
                parameter_injection_layer_num: int, decoder_layer_num: int, output_layer_num: int, device_messages: List[Dict[str, Any]],
                performance_num: int, adj_mask: np.ndarray):
        
        super().__init__(device_messages)
        assert hidden_dim % num_heads == 0, (f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.device_level_one_hot_tensors: torch.Tensor
        self.performance_metric = nn.Parameter(torch.empty(performance_num, hidden_dim))
        self.global_token = nn.Parameter(torch.empty(1,hidden_dim))
        attn_mask = self._get_attn_mask(torch.as_tensor(adj_mask, dtype=torch.float32).clone())
        pr_attn_mask = self._get_pr_attn_mask(adj_mask.shape[1])
        self.register_buffer("attn_mask", attn_mask)
        self.register_buffer("pr_attn_mask", pr_attn_mask)
        self.network = nn.ModuleDict({
            "device_embedding_layer": get_mlp_layer(input_dim=self.device_level_one_hot_tensors.shape[1], hidden_dim=hidden_dim, output_dim=hidden_dim,
                                                dropout=dropout, layer_num=embedding_layer_num),
            "parameter_embedding_layer": get_mlp_layer(input_dim=feature_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout,layer_num=embedding_layer_num),
            "parameter_injection_layer": nn.ModuleList([
                Parameter_Injection_Layer_No_Grad_Test(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout, num_heads=num_heads, 
                                        attn_mask=self.attn_mask, pr_attn_mask=self.pr_attn_mask)
                for _ in range(parameter_injection_layer_num)
                ]),
            "decoder": nn.ModuleList([
                Decoder(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout, num_heads=num_heads) for _ in range(decoder_layer_num)
            ]),
            "output_layer": get_mlp_layer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout, layer_num=output_layer_num)
        })
        self._init_weight()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        B, _, _ = x.shape
        final_device_one_hot_tensors = self.network['device_embedding_layer'](self.device_level_one_hot_tensors.unsqueeze(0).expand(B, -1, -1))
        final_global_tensors = self.global_token.reshape(1, 1, -1).expand(B ,-1, -1)
        device_embedding_tensors = torch.cat((final_device_one_hot_tensors, final_global_tensors), dim=1)
        parameter_embedding_tensors = self.network['parameter_embedding_layer'](x)
        for sub_layer in self.network['parameter_injection_layer']:
            device_embedding_tensors = sub_layer(device_embedding_tensors, parameter_embedding_tensors)
        parameter_tensors = self.performance_metric.unsqueeze(0). expand(B, -1, -1)
        for sub_layer in self.network['decoder']:
            parameter_tensors, _ = sub_layer(device_embedding_tensors, parameter_tensors)
        output_tensors = self.network['output_layer'](parameter_tensors).reshape(B, -1)
        
        return output_tensors
        
    def _get_attn_mask(self, adj_mask: torch.Tensor) -> torch.Tensor:
        
        N, _ = adj_mask.shape
        row_tensors = torch.ones((1, N), dtype=torch.float32, device=adj_mask.device)
        col_tensors = torch.ones((N+1, 1), dtype=torch.float32, device=adj_mask.device)
        adj_mask = torch.cat((adj_mask, row_tensors), dim=0)
        adj_mask = torch.cat((adj_mask, col_tensors), dim=-1)
        
        return (adj_mask - 1) * 1e9
    
    def _get_pr_attn_mask(self, device_number: int) -> torch.Tensor: 
        
        tmp_mask = torch.eye(device_number, dtype=torch.float32)
        row_tensors = torch.ones((1, device_number), dtype=torch.float32)
        pr_attn_mask = torch.cat((tmp_mask, row_tensors), dim=0)
        
        return (pr_attn_mask - 1) * 1e9
        
    def _init_weight(self, ) -> None:
        
        nn.init.normal_(self.performance_metric, mean=0.0, std=0.02)
        nn.init.normal_(self.global_token, mean=0.0, std=0.02)