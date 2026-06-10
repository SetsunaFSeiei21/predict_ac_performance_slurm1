import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderDetachQKFreezeQK(nn.Module):
    """
    Encoder block for ablation study:

    - Q/K input representations are detached.
    - W_Q and W_K are frozen.
    - V path is not detached, so value/content-path gradients can still flow to x.
    - W_V, W_O, FFN, LayerNorm, residual projection remain trainable.

    Gradient behavior:
        Q -> x: blocked
        K -> x: blocked
        V -> x: kept

        W_Q: frozen
        W_K: frozen
        W_V: trainable
        W_O: trainable
        FFN: trainable
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int,
        dropout: float,
        **kwargs
    ) -> None:
        super().__init__()

        assert input_dim % num_heads == 0, (
            f"input_dim={input_dim} must be divisible by num_heads={num_heads}"
        )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Q/K/V/O projections
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)

        self.dropout = nn.Dropout(dropout)

        # Pre-LN structure
        self.normalize_layer1 = nn.LayerNorm(input_dim)
        self.normalize_layer2 = nn.LayerNorm(input_dim)

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
        )

        self.residual_proj = (
            nn.Identity()
            if output_dim == input_dim
            else nn.Linear(input_dim, output_dim)
        )

        self._init_weight()

        # Freeze W_Q and W_K after initialization.
        self._freeze_qk_projection()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:
                Tensor with shape [batch_size, node_num, input_dim].

            attention_mask:
                Optional attention mask.

                Supported shapes:
                    [node_num, node_num]
                    [batch_size, node_num, node_num]
                    [batch_size * num_heads, node_num, node_num]
                    [batch_size, num_heads, node_num, node_num]

                Bool mask:
                    True means blocked.
                    False means allowed.

                Float mask:
                    Usually 0 for allowed and -inf for blocked.

        Returns:
            x:
                Updated tensor with shape [batch_size, node_num, output_dim].

            attn_score:
                Attention weights with shape [batch_size, num_heads, node_num, node_num].
        """

        # x: [B, N, D]
        batch_size, node_num, _ = x.shape

        # Pre-LayerNorm
        normed_x = self.normalize_layer1(x)

        # Q/K use detached hidden states.
        # This blocks score-path gradients to upstream representation x.
        x_qk = normed_x.detach()

        # W_Q and W_K are frozen.
        # no_grad avoids building useless computation graph for Q/K projection.
        with torch.no_grad():
            q = self.q_proj(x_qk)
            k = self.k_proj(x_qk)

        # V keeps normal gradient path to normed_x and upstream x.
        v = self.v_proj(normed_x)

        # [B, N, D] -> [B, H, N, Dh]
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Attention score: [B, H, N, N]
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn_logits = self._apply_attention_mask(
                attn_logits=attn_logits,
                attention_mask=attention_mask,
            )

        attn_score = F.softmax(attn_logits, dim=-1)
        attn_score = self.dropout(attn_score)

        # Attention output: [B, H, N, Dh]
        attn_out = torch.matmul(attn_score, v)

        # [B, H, N, Dh] -> [B, N, D]
        attn_out = self._merge_heads(attn_out)

        # Output projection remains trainable.
        attn_out = self.out_proj(attn_out)

        # Attention residual.
        attn_out = attn_out + x

        # FFN residual block.
        normed_attn_out = self.normalize_layer2(attn_out)
        ffn_out = self.ffn(normed_attn_out)

        x = ffn_out + self.residual_proj(attn_out)

        return x, attn_score

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert [B, N, D] to [B, H, N, Dh].
        """

        batch_size, node_num, _ = x.shape

        x = x.view(
            batch_size,
            node_num,
            self.num_heads,
            self.head_dim,
        )

        x = x.transpose(1, 2).contiguous()

        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert [B, H, N, Dh] to [B, N, D].
        """

        batch_size, num_heads, node_num, head_dim = x.shape

        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, node_num, num_heads * head_dim)

        return x

    def _apply_attention_mask(
        self,
        attn_logits: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply attention mask to attn_logits.

        Args:
            attn_logits:
                Shape [B, H, N, N].

            attention_mask:
                Supported shapes:
                    [N, N]
                    [B, N, N]
                    [B * H, N, N]
                    [B, H, N, N]

        Returns:
            Masked attention logits with shape [B, H, N, N].
        """

        batch_size, num_heads, node_num, _ = attn_logits.shape

        mask = attention_mask.to(device=attn_logits.device)

        if mask.dim() == 2:
            # [N, N] -> [1, 1, N, N]
            mask = mask.unsqueeze(0).unsqueeze(0)

        elif mask.dim() == 3:
            if mask.shape[0] == batch_size:
                # [B, N, N] -> [B, 1, N, N]
                mask = mask.unsqueeze(1)

            elif mask.shape[0] == batch_size * num_heads:
                # [B * H, N, N] -> [B, H, N, N]
                mask = mask.view(batch_size, num_heads, node_num, node_num)

            else:
                raise ValueError(
                    f"Unsupported 3D attention_mask shape: {tuple(mask.shape)}. "
                    f"Expected [B, N, N] or [B*num_heads, N, N]."
                )

        elif mask.dim() == 4:
            # Expected [B, H, N, N] or broadcastable form.
            pass

        else:
            raise ValueError(
                f"Unsupported attention_mask dim={mask.dim()}, "
                f"shape={tuple(mask.shape)}."
            )

        if mask.dtype == torch.bool:
            # Bool mask: True means blocked.
            attn_logits = attn_logits.masked_fill(mask, float("-inf"))
        else:
            # Float mask: usually 0 for allowed and -inf for blocked.
            mask = mask.to(dtype=attn_logits.dtype)
            attn_logits = attn_logits + mask

        return attn_logits

    def _freeze_qk_projection(self) -> None:
        """
        Freeze W_Q and W_K.
        """

        for param in self.q_proj.parameters():
            param.requires_grad_(False)

        for param in self.k_proj.parameters():
            param.requires_grad_(False)

    def _init_weight(self) -> None:
        """
        Initialize all trainable and frozen parameters.
        Frozen parameters are initialized first and then frozen.
        """

        # Q/K/V/O projection initialization
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)

        if self.k_proj.bias is not None:
            nn.init.zeros_(self.k_proj.bias)

        if self.v_proj.bias is not None:
            nn.init.zeros_(self.v_proj.bias)

        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

        # FFN initialization
        ffn_linear_layers = [
            layer for layer in self.ffn
            if isinstance(layer, nn.Linear)
        ]

        for layer in ffn_linear_layers[:-1]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        last_layer = ffn_linear_layers[-1]
        nn.init.kaiming_uniform_(last_layer.weight, mode="fan_in")

        if last_layer.bias is not None:
            nn.init.zeros_(last_layer.bias)

        # Residual projection initialization
        if isinstance(self.residual_proj, nn.Linear):
            nn.init.xavier_uniform_(self.residual_proj.weight)

            if self.residual_proj.bias is not None:
                nn.init.zeros_(self.residual_proj.bias)

        # LayerNorm initialization
        nn.init.constant_(self.normalize_layer1.weight, 1.0)
        nn.init.constant_(self.normalize_layer1.bias, 0.0)

        nn.init.constant_(self.normalize_layer2.weight, 1.0)
        nn.init.constant_(self.normalize_layer2.bias, 0.0)