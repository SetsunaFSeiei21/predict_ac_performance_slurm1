import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any

from src.layers import GlobalEncoder, Decoder
from src.modules import Structure_Encoding_Layer
from src.utils import get_mlp_layer
from src.base import Device_BaseModel

class Global_Encoder_WITH_SE(Device_BaseModel):
    
    def __init__(self, feature_dim: int, hidden_dim: int, output_dim: int, dropout: float, num_heads: int,
            embedding_layer_num: int, structure_encoding_layer_num: int, encoder_layer_num: int, unmask_encoder_layer_num: int, decoder_layer_num: int, output_layer_num: int,
            performance_num: int, device_messages: List[Dict[str, Any]], adj_mask: np.ndarray) -> None:
        
        assert hidden_dim % num_heads == 0, (f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        assert encoder_layer_num >= unmask_encoder_layer_num, (f"encoder_layer_num={encoder_layer_num} should not smaller than\
                                                            unmask_encoder_layer_num={unmask_encoder_layer_num}")
        super().__init__(device_messages)
        self.device_level_one_hot_tensors: torch.Tensor
        self.unmask_encoder_layer_num = unmask_encoder_layer_num
        self.performance_metric = nn.Parameter(torch.empty(performance_num, hidden_dim))
        self.global_token = nn.Parameter(torch.empty(1, hidden_dim))
        se_attn_mask = self._get_se_attn_mask(torch.as_tensor(adj_mask, dtype=torch.float32).clone())
        self.register_buffer("se_attn_mask", se_attn_mask)
        ge_attn_mask = self._get_ge_attn_mask(torch.as_tensor(adj_mask, dtype=torch.float32).clone())
        self.register_buffer("ge_attn_mask", ge_attn_mask)
        self.network = nn.ModuleDict({
            "device_embedding_layer": get_mlp_layer(input_dim=self.device_level_one_hot_tensors.shape[1], hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout,
                                                    layer_num=embedding_layer_num),
            "parameter_embedding_layer": get_mlp_layer(input_dim=feature_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout, layer_num=embedding_layer_num),
            "structure_encoding_layer": nn.ModuleList([
                Structure_Encoding_Layer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout, num_heads=num_heads, attn_mask=self.se_attn_mask)
                for _ in range(structure_encoding_layer_num)
            ]),
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
        share_global_controller = self.global_token.unsqueeze(0).expand(B, -1, -1)
        device_embeddings = self.network["device_embedding_layer"](final_device_one_hot_tensors)
        parameter_embeddings = self.network["parameter_embedding_layer"](x)
        se_input = torch.cat((device_embeddings, share_global_controller), dim=1)
        for sub_layer in self.network["structure_encoding_layer"]:
            se_input = sub_layer(se_input)
        ge_device_embeddings = se_input[:, :-1, :]
        ge_share_global_encoder = se_input[:, -1:, :]
        for idx, sub_layer in enumerate(self.network["global_encoder_layer"]):
            ge_input_tensors = torch.cat((ge_device_embeddings, ge_share_global_encoder), dim=1)
            if idx < len(self.network["global_encoder_layer"]) - self.unmask_encoder_layer_num:
                ge_input_tensors, _ = sub_layer(ge_input_tensors, self.ge_attn_mask)
            else:
                ge_input_tensors, _ = sub_layer(ge_input_tensors)
            tmp_tensors = ge_input_tensors[:, :-1, :]
            ge_device_embeddings = tmp_tensors + parameter_embeddings
        performance_tensors = self.performance_metric.unsqueeze(0).expand(B, -1, -1)
        for sub_layer in self.network["decoder_layer"]:
            performance_tensors, _ = sub_layer(ge_device_embeddings, performance_tensors)
        output_tensors = self.network["output_layer"](performance_tensors).reshape(B, -1)
        
        return output_tensors
        
    def _get_se_attn_mask(self, adj_mask: torch.Tensor) -> torch.Tensor:
        
        N, _ = adj_mask.shape
        row_tensors = torch.ones((1, N), dtype=torch.float32, device=adj_mask.device)
        col_tensors = torch.ones((N+1, 1), dtype=torch.float32, device=adj_mask.device)
        tmp_attn_mask = torch.cat((adj_mask, row_tensors), dim=0)
        attn_mask = torch.cat((tmp_attn_mask, col_tensors), dim=-1)
        
        return (attn_mask - 1) * 1e9
    
    def _get_ge_attn_mask(self, adj_mask: torch.Tensor) -> torch.Tensor:
        
        N, _ = adj_mask.shape
        col_tensors = torch.zeros((N, 1), dtype=torch.float32, device=adj_mask.device)
        row_tensors = torch.ones((1, N+1), dtype=torch.float32, device=adj_mask.device)
        tmp_attn_mask = torch.cat((adj_mask, col_tensors), dim=-1)
        attn_mask = torch.cat((tmp_attn_mask, row_tensors), dim=0)
        
        return (attn_mask - 1) * 1e9
    
    def _init_weight(self,) -> None:
        
        nn.init.normal_(self.global_token, mean=0.0, std=0.02)
        nn.init.normal_(self.performance_metric, mean=0.0, std=0.02)