import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Dict, List, Any, Optional, Tuple

from src.base import Device_BaseModel
from src.utils import get_mlp_layer
from src.layers import Decoder


class _DenseGATSharedNoGradFunction(torch.autograd.Function):
    """
    Custom backward for shared-parameter dense GAT no-grad test.

    Forward:
        h = W x
        score_ijh = LeakyReLU(a_src_h^T h_ih + a_dst_h^T h_jh)
        attention_ijh = softmax_j(score_ijh)
        out_ih = sum_j attention_ijh * h_jh

    Backward:
        Value path:
            all heads preserve value-gradient to W and x.

        Score path:
            only true heads compute score-gradient.

        Proxy heads:
            skip softmax-score backward completely.
            They only keep value-path gradient.

    This follows the latest encoder_no_grad_test idea:
        same forward computation, but restricted backward score-gradient.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        lin_weight: torch.Tensor,
        att_src: torch.Tensor,
        att_dst: torch.Tensor,
        adj: torch.Tensor,
        true_head_num: int,
        heads: int,
        out_channels: int,
        negative_slope: float,
        dropout_p: float,
        training: bool,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [B, N, Fin]

            lin_weight:
                [H * C, Fin]

            att_src / att_dst:
                [1, 1, H, C]

            adj:
                [B, N, N], binary mask.
                adj[i, j] = 1 means target node i can attend to source node j.

        Returns:
            out:
                [B, N, H, C]
        """

        B, N, Fin = x.shape
        H = int(heads)
        C = int(out_channels)
        T = int(true_head_num)
        D = H * C
        true_dim = T * C

        # ------------------------------------------------------------
        # 1. Shared projection
        # ------------------------------------------------------------
        h = F.linear(x, lin_weight, bias=None)
        h = h.view(B, N, H, C)

        # ------------------------------------------------------------
        # 2. Dense additive GAT score
        # ------------------------------------------------------------
        alpha_src = (h * att_src).sum(dim=-1)  # [B, N, H]
        alpha_dst = (h * att_dst).sum(dim=-1)  # [B, N, H]

        score_raw = alpha_src.unsqueeze(2) + alpha_dst.unsqueeze(1)  # [B, N, N, H]
        score_act = F.leaky_relu(score_raw, negative_slope=negative_slope)

        if score_act.dtype in (torch.float16, torch.bfloat16):
            mask_value = -1e4
        else:
            mask_value = -1e9

        score_masked = score_act.masked_fill(adj.unsqueeze(-1) <= 0, mask_value)

        attention = torch.softmax(score_masked, dim=2)

        if training and dropout_p > 0.0:
            keep_prob = 1.0 - dropout_p
            dropout_mask = (
                torch.rand_like(attention) < keep_prob
            ).to(attention.dtype) / keep_prob
            attention_used = attention * dropout_mask
        else:
            dropout_mask = torch.empty(
                0,
                device=x.device,
                dtype=x.dtype,
            )
            attention_used = attention

        # ------------------------------------------------------------
        # 3. Value aggregation
        # ------------------------------------------------------------
        out = torch.einsum(
            "bijh,bjhc->bihc",
            attention_used,
            h,
        )

        # Save only true-head tensors for score-gradient.
        # Proxy heads skip score-gradient in backward.
        if T > 0:
            h_true = h[:, :, :T, :].contiguous()
            score_raw_true = score_raw[:, :, :, :T].contiguous()
            att_src_true = att_src[:, :, :T, :].contiguous()
            att_dst_true = att_dst[:, :, :T, :].contiguous()
        else:
            h_true = torch.empty(
                B,
                N,
                0,
                C,
                device=x.device,
                dtype=x.dtype,
            )
            score_raw_true = torch.empty(
                B,
                N,
                N,
                0,
                device=x.device,
                dtype=x.dtype,
            )
            att_src_true = torch.empty(
                1,
                1,
                0,
                C,
                device=x.device,
                dtype=x.dtype,
            )
            att_dst_true = torch.empty(
                1,
                1,
                0,
                C,
                device=x.device,
                dtype=x.dtype,
            )

        ctx.save_for_backward(
            x,
            h_true,
            score_raw_true,
            attention,
            dropout_mask,
            adj,
            att_src_true,
            att_dst_true,
            lin_weight,
        )

        ctx.heads = H
        ctx.out_channels = C
        ctx.true_head_num = T
        ctx.true_dim = true_dim
        ctx.negative_slope = float(negative_slope)
        ctx.dropout_p = float(dropout_p)
        ctx.training = bool(training)

        return out

    @staticmethod
    def backward(
        ctx,
        grad_out: torch.Tensor,
    ):
        """
        grad_out:
            [B, N, H, C]
        """

        (
            x,
            h_true,
            score_raw_true,
            attention,
            dropout_mask,
            adj,
            att_src_true,
            att_dst_true,
            lin_weight,
        ) = ctx.saved_tensors

        H = ctx.heads
        C = ctx.out_channels
        T = ctx.true_head_num
        true_dim = ctx.true_dim
        negative_slope = ctx.negative_slope
        dropout_p = ctx.dropout_p
        training = ctx.training

        B, N, Fin = x.shape
        D = H * C

        grad_out = grad_out.contiguous()

        if training and dropout_p > 0.0:
            attention_used = attention * dropout_mask
        else:
            attention_used = attention

        # ============================================================
        # 1. Value path for all heads.
        #
        # out_i,h,c = sum_j attention_i,j,h * h_j,h,c
        #
        # d h_j,h,c = sum_i attention_i,j,h * d out_i,h,c
        # ============================================================
        grad_h_value = torch.einsum(
            "bijh,bihc->bjhc",
            attention_used,
            grad_out,
        )  # [B, N, H, C]

        grad_h_value_flat = grad_h_value.reshape(B * N, D)
        x_flat = x.reshape(B * N, Fin)

        grad_lin_weight = grad_h_value_flat.transpose(0, 1).matmul(x_flat)

        # latest encoder_no_grad_test style:
        # hidden gradient only comes from value path.
        grad_x = grad_h_value_flat.matmul(lin_weight).view(B, N, Fin)

        grad_att_src = torch.zeros(
            1,
            1,
            H,
            C,
            device=x.device,
            dtype=x.dtype,
        )
        grad_att_dst = torch.zeros(
            1,
            1,
            H,
            C,
            device=x.device,
            dtype=x.dtype,
        )

        # ============================================================
        # 2. Score path only for true heads.
        #
        # Proxy heads skip:
        #     grad_attention
        #     softmax backward
        #     score backward
        #     att_src / att_dst gradient
        #     score-path gradient to W
        #
        # This is the main optimization.
        # ============================================================
        if T > 0:
            grad_out_true = grad_out[:, :, :T, :]               # [B, N, T, C]
            attention_true = attention[:, :, :, :T]             # [B, N, N, T]

            if training and dropout_p > 0.0:
                dropout_true = dropout_mask[:, :, :, :T]
                attention_used_true = attention_true * dropout_true
            else:
                dropout_true = None
                attention_used_true = attention_true

            # --------------------------------------------------------
            # out_i,h,c = sum_j attention_i,j,h * h_j,h,c
            #
            # d attention_i,j,h = dot(d out_i,h, h_j,h)
            # --------------------------------------------------------
            grad_attention_used_true = torch.einsum(
                "bihc,bjhc->bijh",
                grad_out_true,
                h_true,
            )  # [B, N, N, T]

            if training and dropout_p > 0.0:
                grad_attention_true = grad_attention_used_true * dropout_true
            else:
                grad_attention_true = grad_attention_used_true

            # --------------------------------------------------------
            # attention = softmax(score, dim=source_j)
            # --------------------------------------------------------
            softmax_dot = (
                grad_attention_true * attention_true
            ).sum(dim=2, keepdim=True)

            grad_score_masked_true = attention_true * (
                grad_attention_true - softmax_dot
            )

            # masked_fill derivative:
            # invalid edges should not receive score-gradient.
            grad_score_act_true = grad_score_masked_true * adj.unsqueeze(-1)

            # --------------------------------------------------------
            # LeakyReLU backward
            # --------------------------------------------------------
            leaky_grad = torch.where(
                score_raw_true >= 0,
                torch.ones_like(score_raw_true),
                torch.full_like(score_raw_true, negative_slope),
            )

            grad_score_raw_true = grad_score_act_true * leaky_grad

            # --------------------------------------------------------
            # score_ijh = alpha_src_ih + alpha_dst_jh
            # --------------------------------------------------------
            grad_alpha_src = grad_score_raw_true.sum(dim=2)  # [B, N, T]
            grad_alpha_dst = grad_score_raw_true.sum(dim=1)  # [B, N, T]

            # --------------------------------------------------------
            # alpha_src_ih = dot(h_ih, att_src_h)
            # alpha_dst_jh = dot(h_jh, att_dst_h)
            # --------------------------------------------------------
            grad_att_src_true = (
                grad_alpha_src.unsqueeze(-1) * h_true
            ).sum(dim=(0, 1))  # [T, C]

            grad_att_dst_true = (
                grad_alpha_dst.unsqueeze(-1) * h_true
            ).sum(dim=(0, 1))  # [T, C]

            grad_att_src[:, :, :T, :] = grad_att_src_true.view(1, 1, T, C)
            grad_att_dst[:, :, :T, :] = grad_att_dst_true.view(1, 1, T, C)

            grad_h_score_true = (
                grad_alpha_src.unsqueeze(-1) * att_src_true
                + grad_alpha_dst.unsqueeze(-1) * att_dst_true
            )  # [B, N, T, C]

            # --------------------------------------------------------
            # latest encoder_no_grad_test style:
            # score path updates W for true heads,
            # but does not send gradient back to hidden x.
            #
            # Therefore we add score-gradient to lin_weight only.
            # --------------------------------------------------------
            grad_h_score_true_flat = grad_h_score_true.reshape(
                B * N,
                true_dim,
            )

            x_qk_flat = x.detach().reshape(B * N, Fin)

            grad_lin_score_true = grad_h_score_true_flat.transpose(0, 1).matmul(
                x_qk_flat
            )  # [T*C, Fin]

            grad_lin_weight[:true_dim, :] = (
                grad_lin_weight[:true_dim, :]
                + grad_lin_score_true
            )

        # Return gradients for:
        #
        # x,
        # lin_weight,
        # att_src,
        # att_dst,
        # adj,
        # true_head_num,
        # heads,
        # out_channels,
        # negative_slope,
        # dropout_p,
        # training
        return (
            grad_x,
            grad_lin_weight,
            grad_att_src,
            grad_att_dst,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class DenseGATConv_NoGrad_SharedParam(nn.Module):
    """
    Custom-backward shared-parameter no-grad Dense GAT.

    This replaces the previous PyTorch detach-trick implementation.

    Previous implementation:
        h_value = W x

        true heads:
            h_score = h_value

        proxy heads:
            h_score = h_value.detach()

        This introduces head-wise slice + detach + cat, which can increase
        graph nodes and training time.

    New implementation:
        Forward:
            same dense GAT forward.

        Backward:
            all heads preserve value-path gradient.
            only true heads compute score-gradient.
            proxy heads skip score-gradient completely.

    This follows the optimized encoder_no_grad_test design.
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
        true_head_ratio: float = 0.5,
        proxy_scale: float = 1.0,
        freeze_proxy_attention: bool = False,
        need_attn_score: bool = False,
        safe_self_loop_for_empty_row: bool = True,
    ) -> None:
        super().__init__()

        assert 0.0 <= true_head_ratio <= 1.0, (
            f"true_head_ratio must be in [0, 1], got {true_head_ratio}"
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = float(negative_slope)
        self.dropout = float(dropout)
        self.true_head_ratio = float(true_head_ratio)

        # Kept for interface compatibility with profile_models.py / build_model.
        # In this shared-parameter custom-backward version, proxy_scale is not used.
        self.proxy_scale = float(proxy_scale)

        # Kept for interface compatibility.
        # Latest encoder_no_grad_test-style backward skips proxy score-gradient
        # regardless of this flag.
        self.freeze_proxy_attention = bool(freeze_proxy_attention)

        self.need_attn_score = bool(need_attn_score)
        self.safe_self_loop_for_empty_row = bool(safe_self_loop_for_empty_row)

        self.lin = nn.Linear(
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

        true_head_num = int(round(heads * true_head_ratio))
        true_head_num = max(0, min(heads, true_head_num))

        true_head_mask = torch.zeros(heads, dtype=torch.bool)
        if true_head_num > 0:
            true_head_mask[:true_head_num] = True

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
        nn.init.xavier_uniform_(self.lin.weight)
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
                [N, N] or [B, N, N].
                adj[i, j] = 1 means target node i can attend to source node j.

            mask:
                Optional valid node mask, [B, N] or [N].

            add_loop:
                Whether to add self-loops.

        Returns:
            out:
                [B, N, H * Fout] if concat=True.
                [B, N, Fout] if concat=False.
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

        # If attention score is explicitly requested, use normal dense forward.
        # This path is not optimized for profiling.
        if self.need_attn_score:
            out_h, attention = self._dense_gat_forward_with_attention(
                x=x,
                adj=adj,
                apply_dropout=self.training,
            )
            self.last_attention = attention.detach()
        else:
            self.last_attention = None

            if self.training:
                out_h = _DenseGATSharedNoGradFunction.apply(
                    x,
                    self.lin.weight,
                    self.att_src,
                    self.att_dst,
                    adj,
                    self.true_head_num,
                    self.heads,
                    self.out_channels,
                    self.negative_slope,
                    self.dropout,
                    self.training,
                )
            else:
                # Fast eval path: no custom backward needed.
                out_h, _ = self._dense_gat_forward_with_attention(
                    x=x,
                    adj=adj,
                    apply_dropout=False,
                )

        if self.concat:
            out = out_h.reshape(B, N, self.heads * self.out_channels)
        else:
            out = out_h.mean(dim=2)

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

    def _dense_gat_forward_with_attention(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        apply_dropout: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard dense GAT forward.

        Used for:
            eval mode
            need_attn_score=True

        Not used for optimized training when need_attn_score=False.
        """

        B, N, _ = x.shape

        h = self.lin(x)
        h = h.view(B, N, self.heads, self.out_channels)

        alpha_src = (h * self.att_src).sum(dim=-1)
        alpha_dst = (h * self.att_dst).sum(dim=-1)

        score = alpha_src.unsqueeze(2) + alpha_dst.unsqueeze(1)
        score = F.leaky_relu(score, negative_slope=self.negative_slope)

        if score.dtype in (torch.float16, torch.bfloat16):
            mask_value = -1e4
        else:
            mask_value = -1e9

        score = score.masked_fill(adj.unsqueeze(-1) <= 0, mask_value)

        attention = torch.softmax(score, dim=2)

        if apply_dropout and self.dropout > 0.0:
            attention = F.dropout(
                attention,
                p=self.dropout,
                training=True,
            )

        out = torch.einsum(
            "bijh,bjhc->bihc",
            attention,
            h,
        )

        return out, attention

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
        to avoid softmax over all invalid positions.
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


class GAT_No_Grad_Test(Device_BaseModel):
    """
    GAT top model using custom-backward DenseGATConv_NoGrad_SharedParam.

    Top-level structure is kept the same as src/models/gat.py:

        embedding_layer
        -> graph_conv_layer
        -> residual + norm
        -> FFN
        -> residual + norm
        -> decoder
        -> output_layer

    Only graph_conv_layer is replaced:

        DenseGATConv -> DenseGATConv_NoGrad_SharedParam

    This version follows latest encoder_no_grad_test:

        forward:
            unchanged dense GAT forward.

        backward:
            all heads keep value-gradient.
            only true heads compute score-gradient.
            proxy heads skip score-gradient.
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
        true_head_ratio: float = 0.5,
        proxy_scale: float = 1.0,
        need_attn_score: bool = False,
        freeze_proxy_attention: bool = False,
    ) -> None:

        assert hidden_dim % num_heads == 0, (
            f"hidden_dim:{hidden_dim} must be divisible by num_heads:{num_heads}"
        )

        super().__init__(device_messages)

        self.embedding_layer_num = embedding_layer_num
        self.gat_layer_num = gat_layer_num
        self.decoder_layer_num = decoder_layer_num
        self.output_layer_num = output_layer_num
        self.true_head_ratio = true_head_ratio

        # Kept for interface compatibility.
        # In this shared-param custom-backward version, proxy_scale does not affect backward.
        self.proxy_scale = proxy_scale

        self.need_attn_score = need_attn_score
        self.freeze_proxy_attention = freeze_proxy_attention

        self.performance_metric = nn.Parameter(torch.empty(performance_num, hidden_dim))
        self.register_buffer(
            "adj_tensors",
            torch.as_tensor(adj_mask, dtype=torch.float32),
        )

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
                DenseGATConv_NoGrad_SharedParam(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    concat=True,
                    negative_slope=0.2,
                    dropout=dropout,
                    bias=True,
                    true_head_ratio=true_head_ratio,
                    proxy_scale=proxy_scale,
                    need_attn_score=need_attn_score,
                    freeze_proxy_attention=freeze_proxy_attention,
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
            performance_tensors, _ = sub_layer(
                embedding_tensors,
                performance_tensors,
            )

        output_tensors = self.network["output_layer"](performance_tensors).reshape(B, -1)

        return output_tensors

    def get_true_proxy_head_num(self) -> List[Tuple[int, int]]:
        return [
            conv.get_true_proxy_head_num()
            for conv in self.network["graph_conv_layer"]
        ]

    def get_last_attention(self) -> List[Optional[torch.Tensor]]:
        return [
            conv.last_attention
            for conv in self.network["graph_conv_layer"]
        ]

    def _init_weight(self) -> None:
        nn.init.normal_(self.performance_metric, mean=0.0, std=0.02)