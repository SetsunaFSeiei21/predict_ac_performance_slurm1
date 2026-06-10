import torch
import torch.nn as nn
from src.layers import EncoderDetachQKFreezeQK

class Structure_Encoding_Layer_Detach_qk_wqwk(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float, num_heads: int, attn_mask: torch.Tensor) -> None:
        
        super().__init__()
        self.structure_refining = EncoderDetachQKFreezeQK(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.context_enhancing = EncoderDetachQKFreezeQK(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_heads=num_heads, dropout=dropout)
        self.register_buffer("attn_mask", attn_mask)
        self._init_weight()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x, _ = self.structure_refining(x, self.attn_mask)
        x, _ = self.context_enhancing(x)
        
        return x
        
    def _init_weight(self) -> None:
        self.structure_refining._init_weight()
        self.context_enhancing._init_weight()