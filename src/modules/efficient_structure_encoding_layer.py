import torch
import torch.nn as nn

from src.layers import EfficientEncoder


class Efficient_Structure_Encoding_Layer(nn.Module):
    """
    Efficient version of Structure_Encoding_Layer.

    Original:
        Encoder -> Encoder

    Here:
        EfficientEncoder -> EfficientEncoder
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        num_heads: int,
        attn_mask: torch.Tensor,
        attention_type: str,
        proj_rank: int = 8,
    ) -> None:
        super().__init__()

        num_nodes = int(attn_mask.shape[0])

        self.structure_refining = EfficientEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_type=attention_type,
            num_nodes=num_nodes,
            proj_rank=proj_rank,
        )

        self.context_enhancing = EfficientEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_type=attention_type,
            num_nodes=num_nodes,
            proj_rank=proj_rank,
        )

        self.register_buffer("attn_mask", attn_mask)

        self._init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First encoder uses topology mask in the original code.
        # Here the mask is passed for interface compatibility but ignored
        # inside LinearAttention / LowRankAttention.
        x, _ = self.structure_refining(x, self.attn_mask)

        # Keep the same behavior as the original Structure_Encoding_Layer:
        # second encoder does not use attention mask.
        x, _ = self.context_enhancing(x)

        return x

    def _init_weight(self) -> None:
        self.structure_refining._init_weight()
        self.context_enhancing._init_weight()