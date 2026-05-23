import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any

from src.base import Device_BaseModel
from src.layers import GlobalEncoder, Decoder
from src.utils import get_mlp_layer

class Global_Encoder_WO_SE(Device_BaseModel):
    
    def __init__(self, feature_dim: int, hidden_dim: int, output_dim: int, dropout: float, num_heads: int, 
                embedding_layer_num: int, encoder_layer_num: int, decoder_layer_num: int, output_layer_num: int, performance_num: int, unmask_global_encoder_layer_num: int, 
                device_messages: List[Dict[str, Any]], adj_mask: np.ndarray) -> None:
        
        assert hidden_dim % num_heads == 0, (f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        assert encoder_layer_num >= unmask_global_encoder_layer_num, (f"Encoder_layer_num={encoder_layer_num} should not smaller\
                                                                    than unmask_global_encoder_layer_num={unmask_global_encoder_layer_num}")
        super().__init__(device_messages)
        self.device_level_one_hot_tensors: torch.Tensor
        self.unmask_global_encoder_layer_num = unmask_global_encoder_layer_num
        attn_mask = self._get_attn_mask(torch.as_tensor(adj_mask, dtype=torch.float32).clone())
        self.register_buffer("attn_mask", attn_mask)
        self.global_token = nn.Parameter(torch.empty(1, hidden_dim))
        self.performance_metric = nn.Parameter(torch.empty(performance_num, hidden_dim))
        self.network = nn.ModuleDict({
            "device_embedding_layer": get_mlp_layer(input_dim=self.device_level_one_hot_tensors.shape[1], hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout,
                                                    layer_num=embedding_layer_num),
            "parameter_embedding_layer": get_mlp_layer(input_dim=feature_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout, layer_num=embedding_layer_num),
            "global_encoder_layer": nn.ModuleList([
                GlobalEncoder(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout, num_heads=num_heads) for _ in range(encoder_layer_num)
            ]),
            "decoder_layer": nn.ModuleList([
                Decoder(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout, num_heads=num_heads) for _ in range(decoder_layer_num)
            ]),
            "output_layer": get_mlp_layer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout, layer_num=output_layer_num)
        })
        self._init_weight()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        B, _, _ = x.shape
        final_device_one_hot_tensors = self.device_level_one_hot_tensors.unsqueeze(0).expand(B, -1, -1)
        device_tensors = self.network["device_embedding_layer"](final_device_one_hot_tensors)
        parameter_tensors = self.network["parameter_embedding_layer"](x)
        shared_global_controller = self.global_token.reshape(1, 1, -1).expand(B, -1, -1)
        for idx, sub_layer in enumerate(self.network["global_encoder_layer"]):
            encoder_input = torch.cat((device_tensors, shared_global_controller), dim=1)
            if idx < len(self.network["global_encoder_layer"]) - self.unmask_global_encoder_layer_num:
                encoder_output, _ = sub_layer(encoder_input, self.attn_mask)
            else:
                encoder_output, _ = sub_layer(encoder_input)
            tmp_device_tensors = encoder_output[:,:-1,:]
            device_tensors = tmp_device_tensors + parameter_tensors
        performance_tensors = self.performance_metric.unsqueeze(0).expand(B, -1, -1)
        for sub_layer in self.network["decoder_layer"]:
            performance_tensors, _ = sub_layer(device_tensors, performance_tensors)
        output_tensors = self.network["output_layer"](performance_tensors).reshape(B, -1)
        
        return output_tensors
    
    def _get_attn_mask(self, adj_mask: torch.Tensor) -> torch.Tensor:
        
        N, _ = adj_mask.shape
        col_tensors = torch.ones((N, 1), dtype=torch.float32)
        row_tensors = torch.ones((1, N+1), dtype=torch.float32)
        adj_mask = torch.cat((adj_mask, col_tensors), dim=-1)
        adj_mask = torch.cat((adj_mask, row_tensors), dim=0)
        
        return (adj_mask - 1) * 1e9
    
    def _init_weight(self,) -> None:
        
        nn.init.normal_(self.global_token, mean=0.0, std=0.02)
        nn.init.normal_(self.performance_metric, mean=0.0, std=0.02)