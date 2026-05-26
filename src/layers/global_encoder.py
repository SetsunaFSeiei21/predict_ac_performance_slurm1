import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        num_heads: int,
        **kwargs
    ) -> None:
        super().__init__()

        self.multi_head_attention_layer = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.normalize_layer1 = nn.LayerNorm(normalized_shape=input_dim)
        self.normalize_layer2 = nn.LayerNorm(normalized_shape=input_dim)

        self.ffn = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim, out_features=output_dim),
            nn.Dropout(dropout)
        )

        self.sigmoid_layer = nn.Sigmoid()
        self._init_weight()
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # x: [B, N, D]
        # LayerNorm 不需要 permute
        x = self.normalize_layer1(x)

        past_device_tensor = x[:, :(x.shape[1] - 1), :]

        attn_out, attn_score = self.multi_head_attention_layer(
            query=x,
            key=x,
            value=x,
            attn_mask=attention_mask
        )

        x = x + attn_out

        # LayerNorm 不需要 permute
        x = self.normalize_layer1(x)

        ffn_out = self.ffn(x)
        x = ffn_out + x

        global_device_tensor = x[:, -1, :]

        delta_device_tensors = past_device_tensor + global_device_tensor.unsqueeze(1)

        cosine_similarity = self._cal_cosine_similarity(
            delta_device_tensors,
            x[:, :(x.shape[1] - 1), :]
        )

        # cosine_similarity = self.sigmoid_layer(cosine_similarity)
        cosine_similarity = (cosine_similarity + 1) / 2

        final_device_tensor = (
            (1 - cosine_similarity) * past_device_tensor
            + cosine_similarity * x[:, :(x.shape[1] - 1), :]
        )

        final_output = torch.cat(
            (final_device_tensor, global_device_tensor.unsqueeze(1)),
            dim=1
        )

        return final_output, attn_score
    
    def _cal_cosine_similarity(
        self,
        tensor1: torch.Tensor,
        tensor2: torch.Tensor
    ) -> torch.Tensor:

        return (
            torch.sum(tensor1 * tensor2, dim=-1)
            / (
                torch.sqrt(torch.sum(tensor1 * tensor1, dim=-1))
                * torch.sqrt(torch.sum(tensor2 * tensor2, dim=-1))
                + 1e-8
            )
        ).unsqueeze(-1)
        
    def _init_weight(self) -> None:
        # FFN 初始化
        ffn_linear_layer_index_lst = [
            idx for idx, sub_layer in enumerate(self.ffn)
            if isinstance(sub_layer, nn.Linear)
        ]

        for idx, sub_layer in enumerate(self.ffn):
            if isinstance(sub_layer, nn.Linear):
                if idx != ffn_linear_layer_index_lst[-1]:
                    nn.init.kaiming_uniform_(sub_layer.weight, nonlinearity="relu")
                else:
                    nn.init.kaiming_uniform_(sub_layer.weight, mode="fan_in")

                if sub_layer.bias is not None:
                    nn.init.zeros_(sub_layer.bias)
        
        # MultiheadAttention 初始化
        if hasattr(self.multi_head_attention_layer, "in_proj_weight"):
            nn.init.xavier_uniform_(self.multi_head_attention_layer.in_proj_weight)

            if self.multi_head_attention_layer.in_proj_bias is not None:
                nn.init.zeros_(self.multi_head_attention_layer.in_proj_bias)

        if hasattr(self.multi_head_attention_layer, "out_proj"):
            nn.init.xavier_uniform_(self.multi_head_attention_layer.out_proj.weight)

            if self.multi_head_attention_layer.out_proj.bias is not None:
                nn.init.zeros_(self.multi_head_attention_layer.out_proj.bias)
                
        # LayerNorm 初始化
        nn.init.constant_(self.normalize_layer1.weight, 1.0)
        nn.init.constant_(self.normalize_layer1.bias, 0.0)
        nn.init.constant_(self.normalize_layer2.weight, 1.0)
        nn.init.constant_(self.normalize_layer2.bias, 0.0)


if __name__ == "__main__":
    globalencoder = GlobalEncoder(512, 512, 512, 0.1, 8)