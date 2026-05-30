import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Dict, List, Any, Optional, Tuple

from src.base import Device_BaseModel
from src.utils import get_mlp_layer
from src.layers import Decoder


class DenseGATConv_SplitFull(nn.Module):
    """
    Dense GAT convolution with split score/value projections and full gradients.

    This is the fair full-gradient baseline for DenseGATConv_NoGrad.

    Forward keeps the same GAT-style additive attention form as gat_no_grad_test:

        h_score_i = W_score x_i
        h_value_i = W_value x_i

        e_ij = LeakyReLU(a_src^T h_score_i + a_dst^T h_score_j)
        alpha_ij = softmax_j(e_ij)
        out_i = sum_j alpha_ij h_value_j

    Difference from gat_no_grad_test:
        - All heads receive real score-gradient.
        - No detach is applied to W_score.
        - No proxy-gradient injection is applied.

    Implementation fairness:
        - Score projection is also computed head-by-head, matching gat_no_grad_test.
        - Value weight is also obtained through _build_value_weight_eff().
        - last_attention is detached, matching gat_no_grad_test.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
        safe_self_loop_for_empty_row: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.safe_self_loop_for_empty_row = safe_self_loop_for_empty_row

        self.lin_score = nn.Linear(
            in_features=in_channels,
            out_features=heads * out_channels,
            bias=False,
        )

        self.lin_value = nn.Linear(
            in_features=in_channels,
            out_features=heads * out_channels,
            bias=False,
        )

        self.att_src = nn.Parameter(torch.empty(1, 1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, 1, heads, out_channels))

        if bias:
            if concat:
                self.bias = nn.Parameter(torch.empty(heads * out_channels))
            else:
                self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        # Kept only for implementation symmetry with gat_no_grad_test.
        # All heads are true-gradient heads in this full-gradient baseline.
        true_head_mask = torch.ones(heads, dtype=torch.bool)
        self.register_buffer("true_head_mask", true_head_mask)

        self.last_attention: Optional[torch.Tensor] = None

        self.reset_parameters()

    @property
    def true_head_num(self) -> int:
        return int(self.true_head_mask.sum().item())

    @property
    def proxy_head_num(self) -> int:
        return int((~self.true_head_mask).sum().item())

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin_score.weight)

        # Keep initial forward behavior close to a single-projection GAT.
        with torch.no_grad():
            self.lin_value.weight.copy_(self.lin_score.weight)

        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        add_loop: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [B, N, Fin] or [N, Fin]
            adj:
                [N, N] or [B, N, N], binary adjacency.
                adj[i, j] = 1 means target node i can attend to source node j.
            mask:
                Optional valid node mask, [B, N] or [N].
            add_loop:
                Whether to add self-loops.

        Returns:
            out:
                [B, N, H * Fout] if concat=True
                [B, N, Fout] if concat=False
        """

        squeeze_batch = False

        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_batch = True

        if x.dim() != 3:
            raise ValueError(
                f"Expected x shape [B, N, Fin] or [N, Fin], got {tuple(x.shape)}"
            )

        B, N, _ = x.shape

        adj = self._prepare_adj(
            adj=adj,
            batch_size=B,
            node_num=N,
            device=x.device,
            dtype=x.dtype,
            add_loop=add_loop,
        )

        # ------------------------------------------------------------
        # 1. Score path: full-gradient GAT additive attention
        # ------------------------------------------------------------
        h_score_raw = self._compute_score_projection_by_heads(x)
        h_score = h_score_raw.view(B, N, self.heads, self.out_channels)

        att_src, att_dst = self._build_attention_parameters()

        alpha_src = (h_score * att_src).sum(dim=-1)  # [B, N, H]
        alpha_dst = (h_score * att_dst).sum(dim=-1)  # [B, N, H]

        # score[i, j, h] = score from target node i to source node j.
        # Shape: [B, N, N, H]
        score = alpha_src.unsqueeze(2) + alpha_dst.unsqueeze(1)
        score = F.leaky_relu(score, negative_slope=self.negative_slope)

        score = score.masked_fill(adj.unsqueeze(-1) <= 0, -1e9)

        attention = torch.softmax(score, dim=2)

        if self.training and self.dropout > 0:
            attention = F.dropout(attention, p=self.dropout, training=True)

        self.last_attention = attention.detach()

        # ------------------------------------------------------------
        # 2. Value path: full-gradient value projection
        # ------------------------------------------------------------
        value_weight_eff = self._build_value_weight_eff()

        h_value = F.linear(x, value_weight_eff, bias=None)
        h_value = h_value.view(B, N, self.heads, self.out_channels)

        # out_i = sum_j alpha_ij * h_value_j
        # attention: [B, target_i, source_j, H]
        # h_value:   [B, source_j, H, C]
        # out:       [B, target_i, H, C]
        out = torch.einsum("bijh,bjhc->bihc", attention, h_value)

        if self.concat:
            out = out.reshape(B, N, self.heads * self.out_channels)
        else:
            out = out.mean(dim=2)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)

            if mask.shape != (B, N):
                raise ValueError(
                    f"Expected mask shape [B, N] = {(B, N)}, got {tuple(mask.shape)}"
                )

            out = out * mask.to(device=out.device, dtype=out.dtype).unsqueeze(-1)

        if squeeze_batch:
            out = out.squeeze(0)

        return out

    def _compute_score_projection_by_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute h_score = W_score x head by head.

        This intentionally mirrors gat_no_grad_test's implementation style.
        The only difference is that all heads keep real score-gradient.
        """

        score_weight = self.lin_score.weight
        score_heads = []

        for h in range(self.heads):
            start = h * self.out_channels
            end = (h + 1) * self.out_channels

            weight_h = score_weight[start:end, :]

            # Full-gradient baseline:
            # every head uses normal F.linear, no detach, no torch.no_grad().
            h_score = F.linear(x, weight_h, bias=None)

            score_heads.append(h_score)

        return torch.cat(score_heads, dim=-1)

    def _build_value_weight_eff(self) -> torch.Tensor:
        """
        Full-gradient baseline.

        For implementation symmetry with gat_no_grad_test, value path also
        retrieves its weight through this method.

        Difference from gat_no_grad_test:
            No proxy term is added.
            value_weight_eff == lin_value.weight both in forward and backward.
        """

        return self.lin_value.weight

    def _build_attention_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full-gradient baseline.

        All attention vectors receive normal gradients.
        """

        return self.att_src, self.att_dst

    def _prepare_adj(
        self,
        adj: torch.Tensor,
        batch_size: int,
        node_num: int,
        device: torch.device,
        dtype: torch.dtype,
        add_loop: bool,
    ) -> torch.Tensor:
        """
        Convert adjacency to [B, N, N] binary mask.

        If a row has no valid neighbor, force a self-loop for that row
        to avoid softmax over all -1e9.
        """

        adj = adj.to(device=device, dtype=dtype)

        if adj.dim() == 2:
            if adj.shape != (node_num, node_num):
                raise ValueError(
                    f"Expected adj shape [N, N] = {(node_num, node_num)}, "
                    f"got {tuple(adj.shape)}"
                )
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)

        elif adj.dim() == 3:
            if adj.shape[1:] != (node_num, node_num):
                raise ValueError(
                    f"Expected adj shape [B, N, N] with N={node_num}, "
                    f"got {tuple(adj.shape)}"
                )

            if adj.shape[0] == 1 and batch_size != 1:
                adj = adj.expand(batch_size, -1, -1)
            elif adj.shape[0] != batch_size:
                raise ValueError(
                    f"Expected adj batch size {batch_size}, got {adj.shape[0]}"
                )

        else:
            raise ValueError(
                f"Expected adj shape [N, N] or [B, N, N], got {tuple(adj.shape)}"
            )

        adj = (adj > 0).to(dtype=dtype)

        if add_loop:
            eye = torch.eye(node_num, device=device, dtype=dtype).unsqueeze(0)
            adj = torch.maximum(adj, eye)

        if self.safe_self_loop_for_empty_row:
            row_sum = adj.sum(dim=-1)
            empty_rows = row_sum <= 0

            if empty_rows.any():
                adj = adj.clone()
                b_idx, n_idx = torch.where(empty_rows)
                adj[b_idx, n_idx, n_idx] = 1.0

        return adj

    def get_true_proxy_head_num(self) -> Tuple[int, int]:
        return self.true_head_num, self.proxy_head_num


class GAT_Split_Full(Device_BaseModel):
    """
    GAT top model using DenseGATConv_SplitFull.

    This model is a fair full-gradient baseline for GAT_No_Grad_Test.

    It keeps the same top-level structure as src/models/gat.py and
    src/models/gat_no_grad_test.py:

        embedding_layer
        -> graph_conv_layer
        -> residual + norm
        -> FFN
        -> residual + norm
        -> decoder
        -> output_layer

    The graph layer uses split score/value projections:
        DenseGATConv_SplitFull

    Difference from GAT_No_Grad_Test:
        All heads use full real score-gradient.
        No proxy-gradient approximation is used.
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
    ) -> None:

        assert hidden_dim % num_heads == 0, (
            f"hidden_dim:{hidden_dim} must be divisible by num_heads:{num_heads}"
        )

        super().__init__(device_messages)

        self.embedding_layer_num = embedding_layer_num
        self.gat_layer_num = gat_layer_num
        self.decoder_layer_num = decoder_layer_num
        self.output_layer_num = output_layer_num

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
                DenseGATConv_SplitFull(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    concat=True,
                    negative_slope=0.2,
                    dropout=dropout,
                    bias=True,
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

    def get_true_proxy_head_num(self) -> List[Tuple[int, int]]:
        return [
            conv.get_true_proxy_head_num()
            for conv in self.network["graph_conv_layer"]
        ]

    def _init_weight(self) -> None:
        nn.init.normal_(self.performance_metric, mean=0.0, std=0.02)