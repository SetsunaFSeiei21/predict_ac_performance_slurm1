import torch
import torch.nn as nn
import numpy as np

from typing import Dict, List, Any, Optional, Tuple

from src.base import Device_BaseModel
from src.utils import get_mlp_layer
from src.layers import Decoder
from src.layers.encoder_no_grad_test import HeadwiseTrueGradientProxyKeyAttention


class DenseNoGradGATv2Conv(nn.Module):
    """
    Dense graph masked multi-head attention operator based on
    HeadwiseTrueGradientProxyKeyAttention.

    Purpose:
        Replace torch_geometric.nn.dense.DenseGATConv inside the existing GAT top model.

    Input:
        x:
            [B, N, D]
        adj:
            [N, N] or [B, N, N]

    Output:
        out:
            [B, N, D]

    Note:
        This is not a literal PyG GATv2 implementation:
            e_ij = a^T LeakyReLU(W_s x_i + W_t x_j)

        Instead, it is a dense graph-masked Transformer-style attention operator:
            softmax(QK^T / sqrt(d) + graph_mask) V

        The GATv2-like part is that attention is dynamically computed from
        node features under graph topology masking. The special contribution is
        inherited from HeadwiseTrueGradientProxyKeyAttention:
            - frozen Q
            - detached Q/K hidden path
            - true-gradient heads
            - proxy-gradient heads
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int,
        dropout: float = 0.0,
        concat: bool = True,
        bias: bool = True,
        init_alpha: float = 1.0,
        true_head_ratio: float = 0.5,
        need_attn_score: bool = False,
    ) -> None:
        super().__init__()

        if not concat:
            raise ValueError(
                "DenseNoGradGATv2Conv currently assumes concat=True, "
                "because the original GAT uses DenseGATConv(..., heads=num_heads) "
                "and expects output dimension hidden_dim."
            )

        expected_dim = out_channels * heads
        if in_channels != expected_dim:
            raise ValueError(
                f"Expected in_channels == out_channels * heads, "
                f"got in_channels={in_channels}, out_channels={out_channels}, heads={heads}. "
                f"For the current GAT config, use out_channels=hidden_dim//num_heads."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.concat = concat
        self.need_attn_score = need_attn_score

        self.attention = HeadwiseTrueGradientProxyKeyAttention(
            embed_dim=in_channels,
            num_heads=heads,
            dropout=dropout,
            bias=bias,
            init_alpha=init_alpha,
            true_head_ratio=true_head_ratio,
            need_attn_score=need_attn_score,
        )

        self.last_attn_score: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        add_loop: bool = False,
    ) -> torch.Tensor:
        """
        Keep the call signature close to DenseGATConv:

            conv(x, adj=self.adj_tensors, add_loop=False)

        x:
            [B, N, D]
        adj:
            [N, N] or [B, N, N]
            binary adjacency, where 1 means allowed edge and 0 means masked edge.
            Additive masks with negative values are also accepted.
        add_loop:
            Whether to allow self-attention on diagonal positions.

        Returns:
            [B, N, D]
        """

        squeeze_batch = False
        if x.dim() == 2:
            # For robustness only. Current repo normally uses [B, N, D].
            x = x.unsqueeze(0)
            squeeze_batch = True

        if x.dim() != 3:
            raise ValueError(
                f"Expected x shape [B, N, D] or [N, D], got {tuple(x.shape)}"
            )

        attention_mask = self._build_additive_graph_mask(
            adj=adj,
            x=x,
            add_loop=add_loop,
        )

        out, attn_score = self.attention(
            x=x,
            attention_mask=attention_mask,
        )

        self.last_attn_score = attn_score

        if squeeze_batch:
            out = out.squeeze(0)

        return out

    def _build_additive_graph_mask(
        self,
        adj: torch.Tensor,
        x: torch.Tensor,
        add_loop: bool,
    ) -> torch.Tensor:
        """
        Convert graph adjacency into additive attention mask.

        Why convert manually?
            Encoder_No_Grad_Test internally treats binary masks as:
                1 allowed, 0 blocked,
            and automatically adds self-loops.

            But the original GAT calls:
                DenseGATConv(..., add_loop=False)

            Therefore, to preserve the original behavior, we pass an additive
            mask directly:
                0      -> allowed
                -1e9   -> blocked

            Additive masks bypass the internal self-loop addition logic.
        """

        mask = adj.to(device=x.device)

        # Already additive mask: allowed positions are usually 0,
        # blocked positions are negative large values.
        if torch.is_floating_point(mask) and torch.min(mask) < 0:
            if mask.dim() not in (2, 3):
                raise ValueError(
                    f"Unsupported additive adj shape: {tuple(mask.shape)}. "
                    "Expected [N, N] or [B, N, N]."
                )

            if add_loop:
                mask = mask.clone()
                if mask.dim() == 2:
                    n = mask.shape[0]
                    idx = torch.arange(n, device=mask.device)
                    mask[idx, idx] = 0.0
                else:
                    _, n, _ = mask.shape
                    idx = torch.arange(n, device=mask.device)
                    mask[:, idx, idx] = 0.0

            return mask.to(dtype=x.dtype)

        # Binary adjacency mask.
        mask = mask.float()

        if mask.dim() == 2:
            n, m = mask.shape
            if n != m:
                raise ValueError(
                    f"Expected square adj [N, N], got {tuple(mask.shape)}"
                )

            if add_loop:
                eye = torch.eye(n, device=mask.device, dtype=mask.dtype)
                mask = torch.maximum(mask, eye)

            additive_mask = (1.0 - mask) * -1e9
            return additive_mask.to(dtype=x.dtype)

        if mask.dim() == 3:
            b, n, m = mask.shape
            if n != m:
                raise ValueError(
                    f"Expected square batched adj [B, N, N], got {tuple(mask.shape)}"
                )

            if add_loop:
                eye = torch.eye(n, device=mask.device, dtype=mask.dtype).unsqueeze(0)
                mask = torch.maximum(mask, eye)

            additive_mask = (1.0 - mask) * -1e9
            return additive_mask.to(dtype=x.dtype)

        raise ValueError(
            f"Unsupported adj shape: {tuple(mask.shape)}. "
            "Expected [N, N] or [B, N, N]."
        )

    def restore_frozen_q(self) -> None:
        self.attention.restore_frozen_q()

    def check_q_change(self) -> Tuple[float, float]:
        return self.attention.check_q_change()

    def get_alpha(self) -> List[float]:
        return self.attention.alpha.detach().cpu().tolist()

    def get_true_head_num(self) -> int:
        return self.attention.true_head_num

    def get_proxy_head_num(self) -> int:
        return self.attention.proxy_head_num


class GAT_No_Grad_Test(Device_BaseModel):
    """
    GAT top model with DenseGATConv replaced by DenseNoGradGATv2Conv.

    Compared with src/models/gat.py:
        Original:
            DenseGATConv -> residual + norm -> FFN -> residual + norm

        New:
            DenseNoGradGATv2Conv -> residual + norm -> FFN -> residual + norm

    This keeps the top-level architecture almost unchanged.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        embedding_layer_num: int,
        gat_layer_num: int,
        num_heads: int,
        decoder_layer_num: int,
        output_layer_num: int,
        performance_num: int,
        device_messages: List[Dict[str, Any]],
        adj_mask: np.ndarray,
        init_alpha: float = 1.0,
        true_head_ratio: float = 0.5,
        need_attn_score: bool = False,
    ) -> None:

        assert hidden_dim % num_heads == 0, (
            f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
        )

        super().__init__(device_messages)

        self.embedding_layer_num = embedding_layer_num
        self.gat_layer_num = gat_layer_num
        self.decoder_layer_num = decoder_layer_num
        self.output_layer_num = output_layer_num
        self.num_heads = num_heads
        self.true_head_ratio = true_head_ratio
        self.need_attn_score = need_attn_score

        self.performance_metric = nn.Parameter(torch.empty(performance_num, hidden_dim))
        self.register_buffer("adj_tensors", torch.as_tensor(adj_mask, dtype=torch.float32))

        self.device_level_one_hot_tensors: torch.Tensor

        self.network = nn.ModuleDict({
            "embedding_layer": get_mlp_layer(
                input_dim=(feature_dim + self.device_level_one_hot_tensors.shape[1]),
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout,
                layer_num=embedding_layer_num,
            ),

            "graph_conv_layer": nn.ModuleList([
                DenseNoGradGATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                    bias=True,
                    init_alpha=init_alpha,
                    true_head_ratio=true_head_ratio,
                    need_attn_score=need_attn_score,
                )
                for _ in range(gat_layer_num)
            ]),

            "ffn_layer": nn.ModuleList([
                get_mlp_layer(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    dropout=dropout,
                    layer_num=2,
                )
                for _ in range(gat_layer_num)
            ]),

            "norm_layer1": nn.ModuleList([
                nn.LayerNorm(hidden_dim)
                for _ in range(gat_layer_num)
            ]),

            "norm_layer2": nn.ModuleList([
                nn.LayerNorm(hidden_dim)
                for _ in range(gat_layer_num)
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

        x = torch.cat((x, final_device_level_one_hot_tensors), dim=-1)
        embedding_tensors = self.network["embedding_layer"](x)

        for conv, ffn, norm1, norm2 in zip(
            self.network["graph_conv_layer"],
            self.network["ffn_layer"],
            self.network["norm_layer1"],
            self.network["norm_layer2"],
        ):
            residual = embedding_tensors

            embedding_tensors = conv(
                embedding_tensors,
                adj=self.adj_tensors,
                add_loop=False,
            )

            embedding_tensors = norm1(embedding_tensors + residual)

            residual = embedding_tensors
            embedding_tensors = ffn(embedding_tensors)
            embedding_tensors = norm2(embedding_tensors + residual)

        performance_tensors = self.performance_metric.unsqueeze(0).expand(B, -1, -1)

        for sub_layer in self.network["decoder"]:
            performance_tensors, _ = sub_layer(embedding_tensors, performance_tensors)

        output_tensors = self.network["output_layer"](performance_tensors).reshape(B, -1)

        return output_tensors

    def restore_frozen_q(self) -> None:
        for conv in self.network["graph_conv_layer"]:
            conv.restore_frozen_q()

    def check_q_change(self) -> List[Tuple[float, float]]:
        return [
            conv.check_q_change()
            for conv in self.network["graph_conv_layer"]
        ]

    def get_alpha(self) -> List[List[float]]:
        return [
            conv.get_alpha()
            for conv in self.network["graph_conv_layer"]
        ]

    def get_true_proxy_head_num(self) -> List[Tuple[int, int]]:
        return [
            (conv.get_true_head_num(), conv.get_proxy_head_num())
            for conv in self.network["graph_conv_layer"]
        ]

    def _init_weight(self) -> None:
        nn.init.normal_(self.performance_metric, mean=0.0, std=0.02)