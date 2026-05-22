import torch
import torch.nn as nn
from src.base import Device_BaseModel
from src.layers import Decoder
from src.utils import get_mlp_layer
from typing import Dict, List, Any

class MLP(Device_BaseModel):
    
    def __init__(self, feature_dim: int, hidden_dim: int, output_dim: int, dropout: float,
                embedding_layer_num: int, num_heads: int, decoder_layer_num: int, output_layer_num: int, 
                performance_num: int, device_messages: List[Dict[str, Any]]):
        
        assert hidden_dim % num_heads == 0, ("hidden_dim is not an integer multiple of num_heads.")
        assert decoder_layer_num > 0, ("Decoder_layer_num must larger than 0.")
        super().__init__(device_messages)
        self.embedding_layer_num = embedding_layer_num
        self.decoder_layer_num = decoder_layer_num
        self.output_layer_num = output_layer_num
        self.performance_metric = nn.Parameter(torch.empty(performance_num, hidden_dim))
        self.network = nn.ModuleDict({
            "embedding_layer": get_mlp_layer(input_dim=(feature_dim + self.device_level_one_hot_tensors.shape[1]), 
                                            hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout, layer_num=embedding_layer_num), 
            "decoder": nn.ModuleList([
                Decoder(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim = hidden_dim, dropout=dropout,
                        num_heads=num_heads) for _ in range(decoder_layer_num)
            ]),
            "output_layer": get_mlp_layer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout,
                                        layer_num=output_layer_num)
        })
        self._init_weight()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        B, _, _ = x.shape
        final_device_level_one_hot_tensors = self.device_level_one_hot_tensors.unsqueeze(0).expand(B, -1, -1)
        x = torch.concat((x, final_device_level_one_hot_tensors), dim=-1)
        embedding_tensors = self.network['embedding_layer'](x)
        performance_tensors = self.performance_metric.unsqueeze(0).expand(B, -1, -1)
        for sub_decoder in self.network['decoder']:
            performance_tensors, _ = sub_decoder(embedding_tensors, performance_tensors)
        output_tensors = self.network['output_layer'](performance_tensors)
        final_output = output_tensors.reshape(B, -1)
        
        return final_output
        
    def _init_weight(self) -> None:
        
        nn.init.normal_(self.performance_metric, mean=0.0, std=0.05)
        
    def _initial_weight(self, freeze_direction: int, freeze_layer_num: int) -> None:
        assert freeze_layer_num <= self.embedding_layer_num + self.decoder_layer_num + self.output_layer_num, (
            "freeze_layer_num larger than all layers num!"
        )

        if freeze_direction == 0:
            modules = list(self.modules())
        elif freeze_direction == 1:
            modules = list(reversed(list(self.modules())))
        else:
            raise ValueError("freeze_direction should only be 0 or 1!")

        reset_count = 0

        for layer in modules:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                reset_count += 1

            elif isinstance(layer, Decoder):
                layer._init_weight()
                reset_count += 1

            if reset_count >= freeze_layer_num:
                break