import torch
import torch.nn as nn

from src.layers import ResLayer

def get_mlp_layer(input_dim: int, hidden_dim: int, output_dim: int, dropout: float, layer_num: int):
    
    assert layer_num > 0, ("layer_num must larger than 0")
    layer_lst = []
    if layer_num == 1:
        return nn.Sequential(*[nn.Linear(input_dim, output_dim)])
    else:
        for i in range(layer_num):
            if i == 0:
                layer_lst.append(nn.Linear(input_dim, hidden_dim))
            elif i == layer_num - 1:
                layer_lst.append(nn.Linear(hidden_dim, output_dim))
                return nn.Sequential(*layer_lst)
            else:
                layer_lst.append(nn.Linear(hidden_dim, hidden_dim))
            layer_lst.append(nn.ReLU())
            layer_lst.append(nn.Dropout(dropout))
            
def get_res_mlp_layer(input_dim: int, hidden_dim: int, output_dim: int, dropout: float, block_num: int):
    
    assert block_num >= 0, ("Res_MLP block num should not less than 0")
    if block_num == 0:
        return nn.Sequential(*[nn.Linear(input_dim, output_dim), nn.Dropout(dropout)])
    elif block_num >= 1:
        layer_lst = []
        layer_lst.append(nn.Linear(input_dim, hidden_dim))
        layer_lst.append(nn.ReLU())
        layer_lst.append(nn.Dropout(dropout))
        for _ in range(block_num):
            layer_lst.append(ResLayer(hidden_dim = hidden_dim, hidden_layer_num = 2, dropout = dropout))
        layer_lst.append(nn.Linear(hidden_dim, output_dim))
        layer_lst.append(nn.Dropout(dropout))
        return nn.Sequential(*layer_lst)