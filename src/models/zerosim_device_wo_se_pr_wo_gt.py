import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any

from src.layers import Decoder
from src.modules import Parameter_Injection_Layer
from src.base import Device_BaseModel
from src.utils import get_mlp_layer

class Zerosim_Device_WO_SE_PR_WO_GT(Device_BaseModel):
    
    def __init__(self, feature_dim: int, hidden_dim: int, output_dim: int, dropout: float, num_heads: int, 
                embedding_layer_num: int, parameter_injection_layer_num: int, decoder_layer_num: int, output_layer_num: int, 
                performance_num: int, device_messages: List[Dict[str, Any]], adj_mask: np.ndarray) -> None:
        
        assert hidden_dim % num_heads == 0, (f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        super().__init__(device_messages)
        self.device_level_one_hot_tensors: torch.Tensor
        self.register_buffer("attn_mask", (torch.as_tensor(adj_mask, dtype=torch.float32).clone() - 1) * 1e9)
        pr_attn_mask = self._get_pr_attn_mask(adj_mask.shape[0])
        self.register_buffer("pr_attn_mask", pr_attn_mask)
        self.performance_metric = nn.Parameter(torch.empty(performance_num, hidden_dim))
        self.network = nn.ModuleDict({
            "device_embedding_layer": get_mlp_layer(input_dim=self.device_level_one_hot_tensors.shape[1], hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout, 
                                                layer_num=embedding_layer_num),
            "parameter_embedding_layer": get_mlp_layer(input_dim=feature_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout, layer_num=embedding_layer_num),
            "parameter_injection_layer": nn.ModuleList([
                Parameter_Injection_Layer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout, num_heads=num_heads, 
                                        attn_mask=self.attn_mask, pr_attn_mask=self.pr_attn_mask) for _ in range(parameter_injection_layer_num)
            ]),
            "decoder_layer": nn.ModuleList([
                Decoder(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout, num_heads=num_heads) for _ in range(decoder_layer_num)
            ]),
            "output_layer": get_mlp_layer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout, layer_num=output_layer_num)
        })
        self._init_weight()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        B, _, _ = x.shape
        
        device_embeddings = self.device_level_one_hot_tensors.unsqueeze(0).expand(B, -1, -1)
        device_embeddings = self.network["device_embedding_layer"](device_embeddings)
        
        parameter_embeddings = self.network["parameter_embedding_layer"](x)
        
        for sub_layer in self.network["parameter_injection_layer"]:
            device_embeddings = sub_layer(device_embeddings, parameter_embeddings)
        
        performance_tensors = self.performance_metric.unsqueeze(0).expand(B, -1, -1)
        
        for sub_layer in self.network["decoder_layer"]:
            performance_tensors, _ = sub_layer(device_embeddings, performance_tensors)
        
        output_tensors = self.network["output_layer"](performance_tensors).reshape(B, -1)
        
        return output_tensors
    
    def _get_pr_attn_mask(self, device_num: int) -> torch.Tensor:
        
        return (torch.eye(device_num, dtype=torch.float32) - 1) * 1e9
    
    def _init_weight(self) -> None:
        
        nn.init.normal_(self.performance_metric, mean=0.0, std=0.02)