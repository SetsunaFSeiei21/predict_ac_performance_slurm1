import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any

from src.base import Device_BaseModel
from src.modules import Structure_Encoding_Layer, Parameter_Injection_GE_Layer
from src.layers import Decoder
from src.utils import get_mlp_layer

class Zerosim_DEVICE_PR_WITH_GE(Device_BaseModel):
    
    def __init__(self, feature_dim: int, hidden_dim: int, output_dim: int, dropout: float, num_heads: int, embedding_layer_num: int, structure_encoding_layer_num: int, 
                parameter_injection_layer_num: int, decoder_layer_num: int, output_layer_num: int, performance_num: int, 
                device_messages: List[Dict[str, Any]], adj_mask: np.ndarray) -> None:
        
        super().__init__(device_messages)
        assert hidden_dim % num_heads == 0, (f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.device_level_one_hot_tensors: torch.Tensor
        self.global_token = nn.Parameter(torch.empty(1, hidden_dim))
        self.performance_metric = nn.Parameter(torch.empty(performance_num, hidden_dim))
        se_attn_mask = self._get_se_attn_mask(torch.as_tensor(adj_mask, dtype=torch.float32).clone())
        pr_attn_mask = self._get_pr_attn_mask(torch.as_tensor(adj_mask, dtype=torch.float32).clone())
        self.register_buffer("se_attn_mask", se_attn_mask)
        self.register_buffer("pr_attn_mask", pr_attn_mask)
        self.network = nn.ModuleDict({
            "device_embedding_layer": get_mlp_layer(input_dim=self.device_level_one_hot_tensors.shape[1], hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout,
                                                    layer_num=embedding_layer_num),
            "parameter_embedding_layer": get_mlp_layer(input_dim=feature_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout, layer_num=embedding_layer_num),
            "structure_encoding_layer": nn.ModuleList([
                Structure_Encoding_Layer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout, num_heads=num_heads, attn_mask=se_attn_mask)
                for _ in range(structure_encoding_layer_num)
            ]),
            "parameter_injection_layer": nn.ModuleList([
                Parameter_Injection_GE_Layer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout, num_heads=num_heads) 
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
        final_device_level_one_hot_tensors = self.device_level_one_hot_tensors.unsqueeze(0).expand(B, -1, -1)
        device_embeddings = self.network["device_embedding_layer"](final_device_level_one_hot_tensors)
        parameter_embeddings = self.network["parameter_embedding_layer"](x)
        share_global_controller = self.global_token.unsqueeze(0).expand(B, -1, -1)
        se_encoder_input = torch.cat((device_embeddings, share_global_controller), dim=1)
        for sub_layer in self.network["structure_encoding_layer"]:
            se_encoder_input = sub_layer(se_encoder_input)
        device_embeddings = se_encoder_input[:, :-1, :]
        share_global_controller = se_encoder_input[:, -1:, :]
        for sub_layer in self.network["parameter_injection_layer"]:
            device_embeddings = sub_layer(device_embeddings, share_global_controller, parameter_embeddings, self.pr_attn_mask)
        performance_tensors = self.performance_metric.unsqueeze(0).expand(B, -1, -1)
        for sub_layer in self.network["decoder"]:
            performance_tensors, _ = sub_layer(device_embeddings, performance_tensors)
        output_tensors = self.network["output_layer"](performance_tensors).reshape(B, -1)
        
        return output_tensors
        
    def _get_se_attn_mask(self, adj_mask: torch.Tensor) -> torch.Tensor:
        
        N, _ = adj_mask.shape
        add_row_tensors = torch.ones((1, N), dtype=torch.float32, device=adj_mask.device)
        add_col_tensors = torch.ones((N+1, 1), dtype=torch.float32, device=adj_mask.device)
        adj_tensors = torch.cat((adj_mask, add_row_tensors), dim=0)
        adj_tensors = torch.cat((adj_tensors, add_col_tensors), dim=-1)
        
        return (adj_tensors - 1) * 1e9
    
    def _get_pr_attn_mask(self, adj_mask: torch.Tensor) -> torch.Tensor:
        
        N, _ = adj_mask.shape
        add_col_tensors = torch.ones((N, 1), dtype=torch.float32, device=adj_mask.device)
        add_row_tensors = torch.ones((1, N+1), dtype=torch.float32, device=adj_mask.device)
        adj_tensors = torch.cat((adj_mask, add_col_tensors), dim=-1)
        adj_tensors = torch.cat((adj_tensors, add_row_tensors), dim=0)
        
        return (adj_tensors - 1) * 1e9
        
    def _init_weight(self,) -> None:
        
        nn.init.normal_(self.global_token, mean=0.0, std=0.02)
        nn.init.normal_(self.performance_metric, mean=0.0, std=0.02)