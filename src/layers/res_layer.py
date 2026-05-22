import torch
import torch.nn as nn

class ResLayer(nn.Module):
    def __init__(self, hidden_dim: int, hidden_layer_num: int, dropout: float, **kwargs):
        super().__init__()
        sub_layer = []
        for i in range(hidden_layer_num):
            sub_layer.append(nn.Linear(hidden_dim, hidden_dim))
            if i < hidden_layer_num - 1:
                sub_layer.append(nn.ReLU())
                sub_layer.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*sub_layer)
        self.final_relu = nn.ReLU()
        self._init_weight()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_relu(self.network(x) + x)
    
    def _init_weight(self):
        for sub_layer in self.network:
            if isinstance(sub_layer, nn.Linear):
                nn.init.kaiming_uniform_(sub_layer.weight, nonlinearity='relu')
                if sub_layer.bias is not None:
                    nn.init.zeros_(sub_layer.bias)