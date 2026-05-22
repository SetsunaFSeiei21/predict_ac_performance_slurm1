import torch
import torch.nn as nn

from src.layers import GlobalEncoder

class Parameter_Injection_GE_Layer(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float, num_heads: int) -> None:
        
        super().__init__()
        self.multi_head_attention_layer1 = GlobalEncoder(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout, num_heads=num_heads)
        self.multi_head_attention_layer2 = GlobalEncoder(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout, num_heads=num_heads)
        self._init_weight()
        
    def forward(self, device_embeddings: torch.Tensor, share_global_controller: torch.Tensor, parameter_embeddings: torch.Tensor, attn_mask: torch.Tensor):
        
        encoder_input_tensors = torch.cat((device_embeddings, share_global_controller), dim=1)
        encoder_input_tensors, _ = self.multi_head_attention_layer1(encoder_input_tensors, attn_mask)
        tmp_device_embeddings = encoder_input_tensors[:, :-1, :]
        device_embeddings = tmp_device_embeddings + parameter_embeddings
        encoder_input_tensors = torch.cat((device_embeddings, share_global_controller), dim=1)
        encoder_input_tensors, _ = self.multi_head_attention_layer2(encoder_input_tensors)
        tmp_device_embeddings = encoder_input_tensors[:, :-1, :]
        device_embeddings = tmp_device_embeddings + parameter_embeddings
        
        return device_embeddings
        
    def _init_weight(self) -> None:
        
        self.multi_head_attention_layer1._init_weight()
        self.multi_head_attention_layer2._init_weight()