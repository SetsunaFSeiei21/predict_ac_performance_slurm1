import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import DenseGATConv
from typing import List, Dict, Any

from src.utils import get_mlp_layer
from src.layers import Decoder
from src.base import Device_BaseModel

class GAT_W_GT(Device_BaseModel):
    
    def __init__(self, feature_dim: int, hidden_dim: int, output_dim: int, dropout: float, num_heads: int, 
                embedding_layer_num: int, gat_layer_num: int, decoder_layer_num: int, output_layer_num: int, performance_num: int,
                device_messages: List[Dict[str, Any]], adj_mask: np.ndarray) -> None:
        
        assert hidden_dim % num_heads == 0, (f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        super().__init__(device_messages)
        self.device_level_one_hot_tensors: torch.Tensor  
        self.performance_metric = nn.Parameter(torch.empty(performance_num, hidden_dim))
        self.global_token = nn.Parameter(torch.empty(1, hidden_dim))
        global_mask = self._get_global_mask(torch.as_tensor(adj_mask, dtype=torch.float32).clone())
        self.register_buffer("global_mask", global_mask)
        self.network = nn.ModuleDict({
            "embedding_layer": get_mlp_layer(input_dim=(feature_dim + self.device_level_one_hot_tensors.shape[1]), hidden_dim=hidden_dim, 
                                            output_dim=hidden_dim, dropout=dropout, layer_num=embedding_layer_num),
            "gat_layer": nn.ModuleList([
                DenseGATConv(in_channels=hidden_dim, out_channels=hidden_dim // num_heads, heads=num_heads, dropout=dropout) for _ in range(gat_layer_num)
            ]),
            "ffn_layer": nn.ModuleList([
                get_mlp_layer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout, layer_num=2)
                for _ in range(gat_layer_num)
            ]), 
            "norm_layer1": nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(gat_layer_num)
            ]),
            "norm_layer2": nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(gat_layer_num)
            ]),
            "decoder": nn.ModuleList([
                Decoder(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout, num_heads=num_heads) 
                for _ in range(decoder_layer_num)
            ]),
            "output_layer": get_mlp_layer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout, 
                                        layer_num=output_layer_num)
        })
        self._init_weight()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        B, _, _ = x.shape
        final_device_one_hot_tensors = self.device_level_one_hot_tensors.unsqueeze(0).expand(B, -1, -1)
        final_global_token = self.global_token.reshape(1,1,-1).expand(B, -1, -1)
        embedding_input = torch.cat((x, final_device_one_hot_tensors), dim=-1)
        embedding_output = self.network["embedding_layer"](embedding_input)
        embedding_output = torch.cat((embedding_output.clone(), final_global_token), dim=1)
        global_adj_mask = self.global_mask.unsqueeze(0).expand(B, -1, -1)
        for conv, ffn, norm1, norm2 in zip(self.network["gat_layer"], self.network["ffn_layer"], self.network["norm_layer1"], self.network["norm_layer2"]):
            residual = embedding_output
            embedding_output = conv(embedding_output, adj = global_adj_mask, add_loop = False)
            embedding_output = norm1(residual + embedding_output)
            residual = embedding_output
            embedding_output = ffn(embedding_output)
            embedding_output = norm2(embedding_output + residual)
        performance_tensors = self.performance_metric.unsqueeze(0).expand(B, -1, -1)
        for sub_layer in self.network['decoder']:
            performance_tensors, _ = sub_layer(embedding_output, performance_tensors)
        output_tensors = self.network['output_layer'](performance_tensors).reshape(B, -1)
        
        return output_tensors
        
    def _get_global_mask(self, adj_mask: torch.Tensor) -> torch.Tensor:
        
        N, _ = adj_mask.shape
        col_tensors = torch.ones((N, 1), dtype=torch.float32)
        row_tensors = torch.ones((1, N+1), dtype=torch.float32)
        tmp_mask = torch.cat((adj_mask, col_tensors), dim=-1)
        global_mask = torch.cat((tmp_mask, row_tensors), dim=0)
        
        return global_mask
    
    def _init_weight(self, ) -> None:
        
        nn.init.normal_(self.global_token, mean=0.0, std=0.02)
        nn.init.normal_(self.performance_metric, mean=0.0, std=0.02)