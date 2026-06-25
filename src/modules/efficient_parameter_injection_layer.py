import torch
import torch.nn as nn

from src.layers import EfficientEncoder, Cross_Attention


class Efficient_Parameter_Injection_Layer(nn.Module):
    """
    Efficient version of Parameter_Injection_Layer.

    Original:
        Encoder -> Cross_Attention -> Encoder -> Cross_Attention

    Here:
        EfficientEncoder -> Cross_Attention -> EfficientEncoder -> Cross_Attention

    Cross_Attention is intentionally unchanged to avoid introducing
    too many experimental variables.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        num_heads: int,
        attn_mask: torch.Tensor,
        pr_attn_mask: torch.Tensor,
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

        self.parameter_injection1 = Cross_Attention(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout,
            num_heads=num_heads,
        )

        self.context_enhancing = EfficientEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_type=attention_type,
            num_nodes=num_nodes,
            proj_rank=proj_rank,
        )

        self.parameter_injection2 = Cross_Attention(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            num_heads=num_heads,
        )

        self.register_buffer("attn_mask", attn_mask)
        self.register_buffer("pr_attn_mask", pr_attn_mask)

        self._init_weight()

    def forward(
        self,
        structure_x: torch.Tensor,
        parameter_kv: torch.Tensor,
    ) -> torch.Tensor:
        structure_x, _ = self.structure_refining(
            structure_x,
            attention_mask=self.attn_mask,
        )

        structure_x, _ = self.parameter_injection1(
            structure_x,
            parameter_kv,
            self.pr_attn_mask,
        )

        # Keep the same behavior as the original Parameter_Injection_Layer:
        # context_enhancing does not use attention mask.
        structure_x, _ = self.context_enhancing(structure_x)

        structure_x, _ = self.parameter_injection2(
            structure_x,
            parameter_kv,
            self.pr_attn_mask,
        )

        return structure_x

    def _init_weight(self) -> None:
        self.structure_refining._init_weight()
        self.parameter_injection1._init_weight()
        self.context_enhancing._init_weight()
        self.parameter_injection2._init_weight()