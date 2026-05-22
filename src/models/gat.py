import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any
from torch_geometric.nn.dense import DenseGATConv

from src.base import Device_BaseModel
from src.utils import get_mlp_layer
from src.layers import Decoder

class GAT(Device_BaseModel):
    
    def __init__(self, feature_dim: int, hidden_dim: int, output_dim: int, dropout: float, embedding_layer_num: int, gat_layer_num: int,
                num_heads: int, decoder_layer_num: int, output_layer_num: int, performance_num: int, 
                device_messages: List[Dict[str, Any]], adj_mask: np.ndarray) -> None:
        
        assert hidden_dim % num_heads == 0, (f"hidden_dim:{hidden_dim} must be divisible by num_heads:{num_heads}")
        super().__init__(device_messages)
        self.embedding_layer_num = embedding_layer_num
        self.gat_layer_num = gat_layer_num
        self.decoder_layer_num = decoder_layer_num
        self.output_layer_num = output_layer_num
        self.performance_metric = nn.Parameter(torch.empty(performance_num, hidden_dim))
        self.register_buffer("adj_tensors", torch.as_tensor(adj_mask, dtype=torch.float32))
        self.device_level_one_hot_tensors: torch.Tensor   
        self.network = nn.ModuleDict({
            "embedding_layer": get_mlp_layer(input_dim=(feature_dim + self.device_level_one_hot_tensors.shape[1]), hidden_dim=hidden_dim, 
                                            output_dim=hidden_dim, dropout=dropout, layer_num=embedding_layer_num),
            "graph_conv_layer": nn.ModuleList([
                DenseGATConv(in_channels=hidden_dim, out_channels=hidden_dim // num_heads, heads=num_heads, dropout=dropout) 
                for _ in range(gat_layer_num) 
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
        final_device_level_one_hot_tensors = self.device_level_one_hot_tensors.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat((x, final_device_level_one_hot_tensors), dim=-1)
        embedding_tensors = self.network['embedding_layer'](x)
        for conv, ffn, norm1, norm2 in zip(self.network["graph_conv_layer"], self.network["ffn_layer"], self.network["norm_layer1"], 
                                        self.network["norm_layer2"]):
            residual = embedding_tensors
            embedding_tensors = conv(embedding_tensors, adj = self.adj_tensors, add_loop = False)
            embedding_tensors = norm1(embedding_tensors + residual)
            residual = embedding_tensors
            embedding_tensors = ffn(embedding_tensors)
            embedding_tensors = norm2(embedding_tensors + residual)
        performance_tensors = self.performance_metric.unsqueeze(0).expand(B, -1, -1)
        for sub_layer in self.network['decoder']:
            performance_tensors, _ = sub_layer(embedding_tensors, performance_tensors)
        output_tensors = self.network['output_layer'](performance_tensors).reshape(B, -1)
        
        return output_tensors
        
    def _init_weight(self,) -> None:
        
        nn.init.normal_(self.performance_metric, mean=0.0, std=0.02)