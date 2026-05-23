import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any

from src.base import Device_BaseModel
from src.layers import Decoder
from torch_geometric.nn.dense import DenseGCNConv
from src.utils import get_mlp_layer


class GCN_W_GT(Device_BaseModel):
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        embedding_layer_num: int,
        gcn_layer_num: int,
        num_heads: int,
        decoder_layer_num: int,
        output_layer_num: int,
        performance_num: int,
        device_messages: List[Dict[str, Any]],
        adj_mask: np.ndarray
    ) -> None:
        
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
        )
        
        super().__init__(device_messages)
        
        self.device_level_one_hot_tensors: torch.Tensor
        
        self.embedding_layer_num = embedding_layer_num
        self.gcn_layer_num = gcn_layer_num
        self.decoder_layer_num = decoder_layer_num
        self.output_layer_num = output_layer_num
        
        self.performance_metric = nn.Parameter(torch.empty(performance_num, hidden_dim))
        self.global_token = nn.Parameter(torch.empty(1, hidden_dim))
        
        global_adj_mask = self._get_global_mask(
            torch.as_tensor(adj_mask, dtype=torch.float32).clone()
        )
        self.register_buffer("global_adj_mask", global_adj_mask)
        
        self.network = nn.ModuleDict({
            "embedding_layer": get_mlp_layer(
                input_dim=feature_dim + self.device_level_one_hot_tensors.shape[1],
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout,
                layer_num=embedding_layer_num
            ),
            "graph_conv_layer": nn.ModuleList([
                DenseGCNConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim
                )
                for _ in range(gcn_layer_num)
            ]),
            "norm_layer": nn.ModuleList([
                nn.LayerNorm(hidden_dim)
                for _ in range(gcn_layer_num)
            ]),
            "decoder": nn.ModuleList([
                Decoder(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    dropout=dropout,
                    num_heads=num_heads
                )
                for _ in range(decoder_layer_num)
            ]),
            "output_layer": get_mlp_layer(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=dropout,
                layer_num=output_layer_num
            )
        })
        
        self._init_weight()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        B, _, _ = x.shape
        
        final_device_level_tensors = self.device_level_one_hot_tensors.unsqueeze(0).expand(B, -1, -1)
        final_global_token = self.global_token.reshape(1, 1, -1).expand(B, -1, -1)
        
        embedding_input = torch.cat((x, final_device_level_tensors), dim=-1)
        embedding_tensors = self.network["embedding_layer"](embedding_input)
        
        embedding_tensors = torch.cat((embedding_tensors, final_global_token), dim=1)
        
        global_adj_mask = self.global_adj_mask.unsqueeze(0).expand(B, -1, -1)
        
        for conv, norm in zip(self.network["graph_conv_layer"], self.network["norm_layer"]):
            residual = embedding_tensors
            
            embedding_tensors = conv(
                embedding_tensors,
                global_adj_mask,
                add_loop=False
            )
            
            embedding_tensors = F.relu(embedding_tensors)
            embedding_tensors = norm(embedding_tensors + residual)
        
        performance_tensors = self.performance_metric.unsqueeze(0).expand(B, -1, -1)
        
        for sub_layer in self.network["decoder"]:
            performance_tensors, _ = sub_layer(embedding_tensors, performance_tensors)
        
        output_tensors = self.network["output_layer"](performance_tensors)
        final_output_tensors = output_tensors.reshape(B, -1)
        
        return final_output_tensors
        
    def _get_global_mask(self, adj_mask: torch.Tensor) -> torch.Tensor:
        
        N, _ = adj_mask.shape
        
        col_tensors = torch.ones((N, 1), dtype=torch.float32, device=adj_mask.device)
        row_tensors = torch.ones((1, N + 1), dtype=torch.float32, device=adj_mask.device)
        
        tmp_mask = torch.cat((adj_mask, col_tensors), dim=-1)
        global_mask = torch.cat((tmp_mask, row_tensors), dim=0)
        
        return global_mask
        
    def _init_weight(self) -> None:
        
        nn.init.normal_(self.performance_metric, mean=0.0, std=0.02)
        nn.init.normal_(self.global_token, mean=0.0, std=0.02)