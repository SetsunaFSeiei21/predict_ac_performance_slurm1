import torch
import torch.nn as nn
from typing import Tuple, Optional


class Encoder_No_Grad_Test(nn.Module):
    
    """_summary_
    测试不同detach策略会造成什么样的影响，甚至是重构attention算子
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int,
        dropout: float,
        freeze_qk_proj: bool = True,
        **kwargs,
    ) -> None:
        
        super().__init__()

        assert input_dim % num_heads == 0, (
            f"input_dim={input_dim} must be divisible by num_heads={num_heads}"
        )

        assert output_dim == input_dim, (
            f"output_dim={output_dim} must equal input_dim={input_dim} "
            "because residual connection uses ffn_out + attn_out."
        )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.freeze_qk_proj = freeze_qk_proj

        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.normalize_layer1 = nn.LayerNorm(input_dim)
        self.normalize_layer2 = nn.LayerNorm(input_dim)

        self.ffn = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim, out_features=output_dim),
            nn.Dropout(dropout),
        )

        self._init_weight()

        if self.freeze_qk_proj:
            self._register_freeze_qk_hooks()

        # 保存初始 Wq/Wk 和 bq/bk。
        # 如果使用 AdamW + weight_decay，并且想严格固定 Wq/Wk，
        # 需要在 optimizer.step() 后调用 restore_frozen_qk()。
        self._save_initial_qk_state()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        mha_mask = self._build_mha_mask(
            attention_mask=attention_mask,
            x=x,
        )

        normed_x = self.normalize_layer1(x)

        # Q/K 输入 detach，V 输入保留梯度。
        # Wq/Wk 的参数梯度会被 hook 置零。
        attn_out, attn_score = self.multi_head_attention(
            query=normed_x.detach(),
            key=normed_x.detach(),
            value=normed_x,
            attn_mask=mha_mask,
            need_weights=True,
            average_attn_weights=False,
        )

        attn_out = attn_out + x

        normed_attn_out = self.normalize_layer2(attn_out)
        ffn_out = self.ffn(normed_attn_out)
        x = ffn_out + attn_out
        
        return x, attn_score

    def _register_freeze_qk_hooks(self) -> None:
        """
        nn.MultiheadAttention uses packed parameters:

            in_proj_weight: [3D, D]
                [0:D]     -> Wq
                [D:2D]    -> Wk
                [2D:3D]   -> Wv

            in_proj_bias: [3D]
                [0:D]     -> bq
                [D:2D]    -> bk
                [2D:3D]   -> bv

        This hook zeros gradients of Wq/Wk and bq/bk.
        Wv, output projection, FFN, and LayerNorm remain trainable.
        """

        D = self.input_dim

        def zero_qk_weight_grad(grad: torch.Tensor) -> torch.Tensor:
            grad = grad.clone()
            grad[:2 * D, :] = 0.0
            return grad

        self.multi_head_attention.in_proj_weight.register_hook(
            zero_qk_weight_grad
        )

        if self.multi_head_attention.in_proj_bias is not None:
            def zero_qk_bias_grad(grad: torch.Tensor) -> torch.Tensor:
                grad = grad.clone()
                grad[:2 * D] = 0.0
                return grad

            self.multi_head_attention.in_proj_bias.register_hook(
                zero_qk_bias_grad
            )

    def _save_initial_qk_state(self) -> None:
        """
        保存初始 Wq/Wk 和 bq/bk。
        如果 optimizer 是 AdamW 且 weight_decay > 0，
        仅仅把梯度置零还不能阻止 Wq/Wk 被 weight decay 改变。
        严格冻结时，在 optimizer.step() 后调用 restore_frozen_qk()。
        """

        D = self.input_dim

        with torch.no_grad():
            self.register_buffer(
                "_initial_qk_weight",
                self.multi_head_attention.in_proj_weight[:2 * D, :].detach().clone(),
            )

            if self.multi_head_attention.in_proj_bias is not None:
                self.register_buffer(
                    "_initial_qk_bias",
                    self.multi_head_attention.in_proj_bias[:2 * D].detach().clone(),
                )
            else:
                self._initial_qk_bias = None

    @torch.no_grad()
    def restore_frozen_qk(self) -> None:
        """
        严格恢复 Wq/Wk 和 bq/bk 到初始化状态。
        如果使用 AdamW + weight_decay，并且想让 Wq/Wk 完全不变，
        需要每次 optimizer.step() 后调用。
        """

        D = self.input_dim

        self.multi_head_attention.in_proj_weight[:2 * D, :].copy_(
            self._initial_qk_weight.to(self.multi_head_attention.in_proj_weight.device)
        )

        if self.multi_head_attention.in_proj_bias is not None and self._initial_qk_bias is not None:
            self.multi_head_attention.in_proj_bias[:2 * D].copy_(
                self._initial_qk_bias.to(self.multi_head_attention.in_proj_bias.device)
            )

    @torch.no_grad()
    def check_qk_change(self) -> tuple[float, float]:
        """
        检查当前 Wq/Wk 和 bq/bk 相比初始化是否发生变化。
        返回：
            qk_weight_diff_max, qk_bias_diff_max
        """

        D = self.input_dim

        current_qk_weight = self.multi_head_attention.in_proj_weight[:2 * D, :]
        qk_weight_diff_max = (
            current_qk_weight - self._initial_qk_weight.to(current_qk_weight.device)
        ).abs().max().item()

        if self.multi_head_attention.in_proj_bias is not None and self._initial_qk_bias is not None:
            current_qk_bias = self.multi_head_attention.in_proj_bias[:2 * D]
            qk_bias_diff_max = (
                current_qk_bias - self._initial_qk_bias.to(current_qk_bias.device)
            ).abs().max().item()
        else:
            qk_bias_diff_max = 0.0

        return qk_weight_diff_max, qk_bias_diff_max

    @torch.no_grad()
    def check_qk_distribution(self) -> tuple[float, float, float, float]:
        """
        检查 Wq/Wk 的均值和标准差。
        返回：
            Wq_mean, Wq_std, Wk_mean, Wk_std
        """

        D = self.input_dim

        Wq = self.multi_head_attention.in_proj_weight[:D, :]
        Wk = self.multi_head_attention.in_proj_weight[D:2 * D, :]

        return (
            Wq.mean().item(),
            Wq.std().item(),
            Wk.mean().item(),
            Wk.std().item(),
        )

    def _build_mha_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        x: torch.Tensor,
    ) -> Optional[torch.Tensor]:

        if attention_mask is None:
            return None

        mask = attention_mask.to(device=x.device)

        # Case 1: already additive mask, e.g., 0 / -1e9
        if torch.is_floating_point(mask) and torch.min(mask) < 0:
            if mask.dim() == 2:
                return mask.to(dtype=x.dtype)

            if mask.dim() == 3:
                return mask.repeat_interleave(self.num_heads, dim=0).to(dtype=x.dtype)

            raise ValueError(
                f"Unsupported additive attention_mask shape: {attention_mask.shape}. "
                "Expected [N, N] or [B, N, N]."
            )

        # Case 2: binary mask, 1 means allowed and 0 means blocked
        if mask.dim() == 2:
            N = mask.shape[0]
            eye = torch.eye(N, device=mask.device, dtype=mask.dtype)
            mask = torch.maximum(mask, eye)

            mha_mask = (1.0 - mask.float()) * -1e9
            return mha_mask.to(dtype=x.dtype)

        if mask.dim() == 3:
            B, N, _ = mask.shape
            eye = torch.eye(N, device=mask.device, dtype=mask.dtype).unsqueeze(0)
            mask = torch.maximum(mask, eye)

            mha_mask = (1.0 - mask.float()) * -1e9
            mha_mask = mha_mask.repeat_interleave(self.num_heads, dim=0)
            return mha_mask.to(dtype=x.dtype)

        raise ValueError(
            f"Unsupported attention_mask shape: {attention_mask.shape}. "
            "Expected [N, N] or [B, N, N]."
        )
        
    def _init_weight(self) -> None:
        # 初始化 FFN
        ffn_linear_index = [
            idx for idx, sub_layer in enumerate(self.ffn)
            if isinstance(sub_layer, nn.Linear)
        ]

        for idx, sub_layer in enumerate(self.ffn):
            if isinstance(sub_layer, nn.Linear):
                if idx != ffn_linear_index[-1]:
                    nn.init.kaiming_uniform_(sub_layer.weight, nonlinearity="relu")
                else:
                    nn.init.kaiming_uniform_(sub_layer.weight, mode="fan_in")

                if sub_layer.bias is not None:
                    nn.init.zeros_(sub_layer.bias)

        # 初始化 MultiheadAttention
        # 这里显式让 Wq 和 Wk 处于不同分布中：
        #   Wq: Xavier Uniform
        #   Wk: Orthogonal
        #   Wv: Xavier Uniform
        if hasattr(self.multi_head_attention, "in_proj_weight"):
            D = self.input_dim

            with torch.no_grad():
                # Wq: Xavier Uniform
                nn.init.xavier_uniform_(
                    self.multi_head_attention.in_proj_weight[:D, :]
                )

                # Wk: Orthogonal，与 Wq 使用不同初始化分布
                nn.init.orthogonal_(
                    self.multi_head_attention.in_proj_weight[D:2 * D, :]
                )

                # Wv: Xavier Uniform
                nn.init.xavier_uniform_(
                    self.multi_head_attention.in_proj_weight[2 * D:3 * D, :]
                )

            if self.multi_head_attention.in_proj_bias is not None:
                nn.init.zeros_(self.multi_head_attention.in_proj_bias)

        if hasattr(self.multi_head_attention, "out_proj"):
            nn.init.xavier_uniform_(self.multi_head_attention.out_proj.weight)

            if self.multi_head_attention.out_proj.bias is not None:
                nn.init.zeros_(self.multi_head_attention.out_proj.bias)

        # 初始化 LayerNorm
        nn.init.constant_(self.normalize_layer1.weight, 1.0)
        nn.init.constant_(self.normalize_layer1.bias, 0.0)
        nn.init.constant_(self.normalize_layer2.weight, 1.0)
        nn.init.constant_(self.normalize_layer2.bias, 0.0)