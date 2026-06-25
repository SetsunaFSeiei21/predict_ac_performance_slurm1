import torch
import torch.nn as nn
import numpy as np

from typing import List, Dict, Any

from src.layers import Decoder, EfficientEncoder
from src.base import Device_BaseModel
from src.utils import get_mlp_layer


class _BaseEfficientFormer(Device_BaseModel):
    """
    ACCFormer-style model with efficient attention in the device-level encoder.

    attention_type:
        "linear"   -> LinearFormer
        "low_rank" -> LowRankFormer
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        num_heads: int,
        embedding_layer_num: int,
        encoder_layer_num: int,
        decoder_layer_num: int,
        output_layer_num: int,
        performance_num: int,
        device_messages: List[Dict[str, Any]],
        attn_mask: np.ndarray,
        attention_type: str,
        proj_rank: int = 8,
    ) -> None:
        super().__init__(device_messages)

        self.embedding_layer_num = embedding_layer_num
        self.encoder_layer_num = encoder_layer_num
        self.decoder_layer_num = decoder_layer_num
        self.output_layer_num = output_layer_num
        self.attention_type = attention_type

        num_nodes = int(attn_mask.shape[0])

        self.performance_metric = nn.Parameter(
            torch.empty(performance_num, hidden_dim)
        )

        # Kept for interface consistency.
        # LinearAttention / LowRankAttention ignore this mask internally.
        self.register_buffer(
            "device_level_attn_mask",
            (torch.tensor(attn_mask) - 1) * 1e9,
        )

        self.network = nn.ModuleDict({
            "embedding_layer": get_mlp_layer(
                input_dim=(
                    feature_dim + self.device_level_one_hot_tensors.shape[1]
                ),
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout,
                layer_num=embedding_layer_num,
            ),
            "encoder": nn.ModuleList([
                EfficientEncoder(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    attention_type=attention_type,
                    num_nodes=num_nodes,
                    proj_rank=proj_rank,
                )
                for _ in range(encoder_layer_num)
            ]),
            "decoder": nn.ModuleList([
                Decoder(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    dropout=dropout,
                    num_heads=num_heads,
                )
                for _ in range(decoder_layer_num)
            ]),
            "output_layer": get_mlp_layer(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=dropout,
                layer_num=output_layer_num,
            ),
        })

        self._init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, _ = x.shape

        final_device_level_one_hot_tensors = (
            self.device_level_one_hot_tensors
            .unsqueeze(0)
            .expand(B, -1, -1)
        )

        x = torch.concat((x, final_device_level_one_hot_tensors), dim=-1)

        embedding_tensors = self.network["embedding_layer"](x)

        for sub_layer in self.network["encoder"]:
            embedding_tensors, _ = sub_layer(
                embedding_tensors,
                self.device_level_attn_mask,
            )

        performance_tensors = self.performance_metric.unsqueeze(0).expand(B, -1, -1)

        for sub_layer in self.network["decoder"]:
            performance_tensors, _ = sub_layer(embedding_tensors, performance_tensors)

        output_tensors = self.network["output_layer"](performance_tensors)
        final_output = output_tensors.reshape(B, -1)

        return final_output

    def _init_weight(self) -> None:
        nn.init.normal_(self.performance_metric, mean=0.0, std=0.02)

class AccFormer_LinearFormer(_BaseEfficientFormer):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["attention_type"] = "linear"
        super().__init__(*args, **kwargs)


class AccFormer_LowRankFormer(_BaseEfficientFormer):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["attention_type"] = "low_rank"
        super().__init__(*args, **kwargs)