import argparse
import csv
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.base import build_model


# ============================================================
# Basic utils
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dummy_device_messages(device_num: int) -> List[Dict[str, str]]:
    kinds = ["NMOS", "PMOS", "RES", "CAP"]
    return [{"kind": kinds[i % len(kinds)]} for i in range(device_num)]


def make_attn_mask(device_num: int, mode: str = "local") -> np.ndarray:
    """
    Same convention as profile_models.py:
        adj[i, j] = 1 means target node i can attend to source node j.

    Important:
        Do NOT use full mask for two-hop coupling experiment.
        If mask is full, every pair is directly connected and there is no
        non-adjacent co-neighbor pair.
    """
    if mode == "full":
        adj = np.ones((device_num, device_num), dtype=np.float32)

    elif mode == "local":
        adj = np.eye(device_num, dtype=np.float32)
        for i in range(device_num):
            if i - 1 >= 0:
                adj[i, i - 1] = 1.0
            if i + 1 < device_num:
                adj[i, i + 1] = 1.0

    elif mode == "random":
        rng = np.random.default_rng(42)
        adj = (rng.random((device_num, device_num)) > 0.7).astype(np.float32)
        adj = np.maximum(adj, adj.T)
        np.fill_diagonal(adj, 1.0)

    else:
        raise ValueError(f"Unknown mask mode: {mode}")

    return adj


def build_zerosim_model(args: argparse.Namespace) -> nn.Module:
    device_messages = make_dummy_device_messages(args.device_num)
    attn_mask = make_attn_mask(args.device_num, args.mask)

    model_hparams = {
        "hidden_dim": args.hidden_dim,
        "output_dim": args.output_dim,
        "dropout": args.dropout,
        "num_heads": args.num_heads,

        "embedding_layer_num": args.embedding_layer_num,
        "encoder_layer_num": args.encoder_layer_num,
        "decoder_layer_num": args.decoder_layer_num,
        "output_layer_num": args.output_layer_num,
        "performance_num": args.performance_num,

        "structure_encoding_layer_num": args.structure_encoding_layer_num,
        "parameter_injection_layer_num": args.parameter_injection_layer_num,

        "gcn_layer_num": args.gcn_layer_num,
        "gat_layer_num": args.gat_layer_num,

        "init_alpha": args.init_alpha,
        "true_head_ratio": args.true_head_ratio,
        "need_attn_score": False,
        "proxy_scale": args.proxy_scale,
        "freeze_proxy_attention": args.freeze_proxy_attention,
    }

    model = build_model(
        model_name=args.model_name,
        model_hyper_parameters=model_hparams,
        input_shape=(args.device_num, args.feature_dim),
        device_messages=device_messages,
        device_level_attn_mask=attn_mask,
    )

    return model


def load_checkpoint_if_needed(model: nn.Module, checkpoint_path: Optional[str]) -> None:
    if checkpoint_path is None:
        print("[Info] No checkpoint provided. Use randomly initialized model.")
        return

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # PyTorch >= 2.6 changed torch.load default weights_only=True.
    # Your checkpoint contains src.dataset.scaler.StandardScaler,
    # so we explicitly set weights_only=False for trusted local checkpoints.
    try:
        ckpt = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        # For older PyTorch versions that do not support weights_only.
        ckpt = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        elif "net" in ckpt:
            state_dict = ckpt["net"]
        else:
            # Some checkpoints are directly state_dict-like.
            state_dict = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint type: {type(ckpt)}")

    # Remove possible DistributedDataParallel prefix.
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    print(f"[Info] Loaded checkpoint: {checkpoint_path}")
    print(f"[Info] Missing keys: {len(missing)}")
    print(f"[Info] Unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("[Info] First 20 missing keys:")
        for k in missing[:20]:
            print(f"  - {k}")

    if len(unexpected) > 0:
        print("[Info] First 20 unexpected keys:")
        for k in unexpected[:20]:
            print(f"  - {k}")


# ============================================================
# ZeroSim partial forward
# ============================================================

@torch.no_grad()
def get_zerosim_hidden_before_target_encoder(
    model: nn.Module,
    x: torch.Tensor,
    parameter_layer_idx: int,
    target_encoder_name: str,
) -> Tuple[torch.Tensor, torch.Tensor, nn.Module, torch.Tensor]:
    """
    Run ZeroSim until the input of a selected Encoder inside one
    Parameter_Injection_Layer.

    Supported target_encoder_name:
        structure_refining
        context_enhancing

    Returns:
        hidden:
            Input hidden representation to selected Encoder.
            Shape: [B, N+1, hidden_dim]

        parameter_tensors:
            Parameter tokens after parameter_embedding_layer.
            Shape: [B, N, hidden_dim]

        target_encoder:
            The selected Encoder module.

        attn_mask:
            The additive attention mask used by this encoder.
            Shape: [N+1, N+1] or None.
    """
    model.eval()

    B, _, _ = x.shape

    device_tensors = (
        model.network["device_embedding_layer"](model.device_level_one_hot_tensors)
        .unsqueeze(0)
        .expand(B, -1, -1)
    )

    global_tensors = model.global_token.unsqueeze(0).expand(B, -1, -1)
    device_tensors = torch.cat([device_tensors, global_tensors], dim=1)

    parameter_tensors = model.network["parameter_embedding_layer"](x)

    # Run structure encoding first.
    for sub_layer in model.network["structure_encoding_layer"]:
        device_tensors = sub_layer(device_tensors)

    # Run previous parameter injection layers.
    for idx, sub_layer in enumerate(model.network["parameter_injection_layer"]):
        if idx == parameter_layer_idx:
            if target_encoder_name == "structure_refining":
                target_encoder = sub_layer.structure_refining
                attn_mask = sub_layer.attn_mask
                hidden = device_tensors

            elif target_encoder_name == "context_enhancing":
                # Need to run structure_refining and parameter_injection1 first.
                tmp, _ = sub_layer.structure_refining(
                    device_tensors,
                    attention_mask=sub_layer.attn_mask,
                )
                tmp, _ = sub_layer.parameter_injection1(
                    tmp,
                    parameter_tensors,
                    sub_layer.pr_attn_mask,
                )
                target_encoder = sub_layer.context_enhancing
                attn_mask = None
                hidden = tmp

            else:
                raise ValueError(
                    f"Unknown target_encoder_name={target_encoder_name}. "
                    "Use structure_refining or context_enhancing."
                )

            return hidden.detach(), parameter_tensors.detach(), target_encoder, attn_mask

        device_tensors = sub_layer(device_tensors, parameter_tensors)

    raise IndexError(
        f"parameter_layer_idx={parameter_layer_idx} is out of range. "
        f"Number of parameter injection layers: {len(model.network['parameter_injection_layer'])}"
    )


# ============================================================
# Manual attention probe for repo Encoder
# ============================================================

def split_qkv_from_mha(
    mha: nn.MultiheadAttention,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reproduce nn.MultiheadAttention Q/K/V projection for self-attention.

    x:
        [B, N, D]
    """
    D = x.shape[-1]

    if mha.in_proj_weight is None:
        raise ValueError("This script expects packed in_proj_weight in MultiheadAttention.")

    w_q, w_k, w_v = mha.in_proj_weight.chunk(3, dim=0)

    if mha.in_proj_bias is not None:
        b_q, b_k, b_v = mha.in_proj_bias.chunk(3, dim=0)
    else:
        b_q = b_k = b_v = None

    q = F.linear(x, w_q, b_q)
    k = F.linear(x, w_k, b_k)
    v = F.linear(x, w_v, b_v)

    return q, k, v


def shape_to_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    [B, N, D] -> [B, H, N, Dh]
    """
    B, N, D = x.shape
    head_dim = D // num_heads

    return (
        x.view(B, N, num_heads, head_dim)
        .transpose(1, 2)
        .contiguous()
    )


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """
    [B, H, N, Dh] -> [B, N, D]
    """
    B, H, N, Dh = x.shape

    return (
        x.transpose(1, 2)
        .contiguous()
        .view(B, N, H * Dh)
    )


def manual_encoder_attention_forward(
    encoder: nn.Module,
    hidden: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Manually run only the attention part of repo Encoder / Encoder_No_Grad_Test.

    Supported:
        1. Standard Encoder:
            encoder.multi_head_attention

        2. Encoder_No_Grad_Test:
            encoder.attention = HeadwiseTrueGradientProxyKeyAttention

    Returns:
        out:
            [B, N, D]

        score:
            [B, H, N, N], retained for gradient measurement.

    Important:
        We detach score and set score.requires_grad_(True) before softmax.
        This isolates dL/dscore measurement and avoids being blocked by
        no_grad Q/K construction in proxy heads.
    """

    # ============================================================
    # Case 1: Standard Encoder with nn.MultiheadAttention
    # ============================================================
    if hasattr(encoder, "multi_head_attention"):
        mha = encoder.multi_head_attention
        num_heads = mha.num_heads
        head_dim = mha.head_dim

        normed_x = encoder.normalize_layer1(hidden)

        q, k, v = split_qkv_from_mha(mha, normed_x)

        q = shape_to_heads(q, num_heads)
        k = shape_to_heads(k, num_heads)
        v = shape_to_heads(v, num_heads)

        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

        if attn_mask is not None:
            mask = attn_mask.to(device=score.device, dtype=score.dtype)
            if mask.dim() == 2:
                score = score + mask.view(1, 1, mask.shape[0], mask.shape[1])
            elif mask.dim() == 3:
                score = score + mask.unsqueeze(1)
            else:
                raise ValueError(f"Unsupported attn_mask shape: {tuple(mask.shape)}")

        # Isolate dL/dscore measurement.
        score = score.detach().requires_grad_(True)
        score.retain_grad()

        attn = torch.softmax(score, dim=-1)

        out_h = torch.matmul(attn, v)
        out = merge_heads(out_h)
        out = mha.out_proj(out)

        return out, score

    # ============================================================
    # Case 2: Encoder_No_Grad_Test with HeadwiseTrueGradientProxyKeyAttention
    # ============================================================
    if hasattr(encoder, "attention"):
        att = encoder.attention

        num_heads = att.num_heads
        head_dim = att.head_dim

        normed_x = encoder.normalize_layer1(hidden)

        # Match no-grad attention forward values:
        # Q/K use detached hidden state.
        x_qk = normed_x.detach()

        with torch.no_grad():
            q = att.q_proj(x_qk)

        # For score-value measurement, compute all K heads in one shot.
        # This gives the same forward K value as the head-wise implementation.
        k_raw = att.k_proj(x_qk)
        alpha_channel = att.alpha.to(device=hidden.device, dtype=hidden.dtype)
        k = k_raw * alpha_channel.view(1, 1, -1)

        # Forward value equals v_proj(normed_x).
        # The proxy-gradient trick only changes backward to Wk/Wv,
        # not the forward V value.
        v = att.v_proj(normed_x)

        q = att._shape_to_heads(q)
        k = att._shape_to_heads(k)
        v = att._shape_to_heads(v)

        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

        if attn_mask is not None:
            mask = attn_mask.to(device=score.device, dtype=score.dtype)

            # Additive mask: 0 / -1e9
            if torch.is_floating_point(mask) and torch.min(mask) < 0:
                if mask.dim() == 2:
                    score = score + mask.view(1, 1, mask.shape[0], mask.shape[1])
                elif mask.dim() == 3:
                    score = score + mask.unsqueeze(1)
                else:
                    raise ValueError(f"Unsupported additive attn_mask shape: {tuple(mask.shape)}")
            else:
                # Binary mask: 1 allowed, 0 blocked.
                if mask.dim() == 2:
                    binary_mask = mask
                    additive_mask = (1.0 - binary_mask.float()) * -1e9
                    score = score + additive_mask.view(1, 1, mask.shape[0], mask.shape[1]).to(score.dtype)
                elif mask.dim() == 3:
                    binary_mask = mask
                    additive_mask = (1.0 - binary_mask.float()) * -1e9
                    score = score + additive_mask.unsqueeze(1).to(score.dtype)
                else:
                    raise ValueError(f"Unsupported binary attn_mask shape: {tuple(mask.shape)}")

        # Isolate dL/dscore measurement.
        # This is important for no-grad attention because proxy heads do not
        # necessarily retain a normal score graph to K/Q parameters.
        score = score.detach().requires_grad_(True)
        score.retain_grad()

        attn_score = torch.softmax(score, dim=-1)

        out_h = torch.matmul(attn_score, v)
        out = att._merge_heads(out_h)
        out = att.out_proj(out)

        return out, score

    raise TypeError(
        "Unsupported encoder type. Expected either encoder.multi_head_attention "
        "or encoder.attention."
    )


# ============================================================
# Triplets and coupling metrics
# ============================================================

def additive_mask_to_adj(
    attn_mask: torch.Tensor,
    exclude_global: bool = True,
) -> torch.Tensor:
    """
    Convert additive mask to binary adjacency.

    Repo convention:
        attn_mask = (adj - 1) * 1e9

    Thus:
        allowed edge has mask = 0
        blocked edge has mask = -1e9
    """
    adj = (attn_mask >= 0).to(torch.float32)

    if exclude_global:
        # ZeroSim appends one global token at the last index.
        # Exclude it from triplet selection to avoid trivial global-token coupling.
        adj = adj.clone()
        adj[-1, :] = 0
        adj[:, -1] = 0

    return adj


def find_two_hop_triplets(
    adj: torch.Tensor,
    max_triplets: int,
    seed: int,
) -> List[Tuple[int, int, int]]:
    """
    Find triplets (i, j, k):

        adj[i, j] = 1
        adj[i, k] = 1
        adj[j, k] = 0
        adj[k, j] = 0
        i, j, k are different

    Meaning:
        j and k are co-neighbors under receiver i,
        but j and k are not directly connected.

        j -> i <- k
    """
    rng = random.Random(seed)

    N = adj.shape[0]
    triplets: List[Tuple[int, int, int]] = []

    for i in range(N):
        neighbors = torch.where(adj[i] > 0)[0].detach().cpu().tolist()
        neighbors = [n for n in neighbors if n != i]

        if len(neighbors) < 2:
            continue

        for j in neighbors:
            for k in neighbors:
                if j == k:
                    continue
                if adj[j, k] > 0 or adj[k, j] > 0:
                    continue
                triplets.append((i, j, k))

    rng.shuffle(triplets)

    if max_triplets > 0:
        triplets = triplets[:max_triplets]

    return triplets


def perturb_hidden_node(
    hidden: torch.Tensor,
    node_idx: int,
    eps: float,
    seed: int,
) -> torch.Tensor:
    """
    Perturb one node hidden representation.

    hidden:
        [1, N, D]
    """
    gen = torch.Generator(device=hidden.device)
    gen.manual_seed(seed)

    out = hidden.detach().clone()

    noise = torch.randn(
        out[:, node_idx, :].shape,
        generator=gen,
        device=hidden.device,
        dtype=hidden.dtype,
    )

    noise = noise / (noise.norm(dim=-1, keepdim=True) + 1e-12)

    scale = out[:, node_idx, :].norm(dim=-1, keepdim=True).clamp_min(1.0)

    out[:, node_idx, :] = out[:, node_idx, :] + eps * scale * noise
    out.requires_grad_(True)

    return out


def effective_score_grad(
    score_grad: torch.Tensor,
    mode: str,
    true_head_num: int,
) -> torch.Tensor:
    """
    mode:
        full:
            use all score gradients.

        latest_custom:
            zero out proxy heads, matching latest custom backward idea:
            proxy heads' score-gradient is not computed.
    """
    if mode == "full":
        return score_grad

    if mode == "latest_custom":
        g = score_grad.clone()
        if true_head_num < g.shape[1]:
            g[:, true_head_num:, :, :] = 0.0
        return g

    raise ValueError(f"Unknown mode: {mode}")


def compute_score_grad_for_target(
    encoder: nn.Module,
    hidden: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    target_i: int,
    mode: str,
    true_head_num: int,
) -> torch.Tensor:
    """
    Local loss:
        L_i = 0.5 * || attention_out_i ||^2

    Returns:
        effective score gradient:
            [B, H, N, N]
    """
    if hidden.grad is not None:
        hidden.grad = None

    out, score = manual_encoder_attention_forward(
        encoder=encoder,
        hidden=hidden,
        attn_mask=attn_mask,
    )

    local_loss = 0.5 * out[:, target_i, :].pow(2).sum()
    local_loss.backward()

    if score.grad is None:
        raise RuntimeError("score.grad is None. score.retain_grad() failed.")

    g = score.grad.detach()

    return effective_score_grad(
        score_grad=g,
        mode=mode,
        true_head_num=true_head_num,
    )


def measure_one_triplet(
    encoder: nn.Module,
    hidden: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    triplet: Tuple[int, int, int],
    mode: str,
    true_head_num: int,
    perturb_eps: float,
    seed: int,
) -> Dict[str, float]:
    """
    Triplet:
        i: receiver / target node
        j: gradient target source node
        k: perturbed co-neighbor source node

    Coupling score:
        C_{k -> j | i}
        =
        || G_{i,j}(hidden + delta_k) - G_{i,j}(hidden) ||
        /
        (|| G_{i,j}(hidden) || + eps)

    where:
        G_{i,j} = dL_i / d score_{i,j}
    """
    i, j, k = triplet

    h0 = hidden.detach().clone().requires_grad_(True)

    g0 = compute_score_grad_for_target(
        encoder=encoder,
        hidden=h0,
        attn_mask=attn_mask,
        target_i=i,
        mode=mode,
        true_head_num=true_head_num,
    )

    h1 = perturb_hidden_node(
        hidden=hidden,
        node_idx=k,
        eps=perturb_eps,
        seed=seed,
    )

    g1 = compute_score_grad_for_target(
        encoder=encoder,
        hidden=h1,
        attn_mask=attn_mask,
        target_i=i,
        mode=mode,
        true_head_num=true_head_num,
    )

    # [H]
    edge_g0 = g0[0, :, i, j]
    edge_g1 = g1[0, :, i, j]

    diff = edge_g1 - edge_g0

    eps = 1e-12
    H = edge_g0.shape[0]
    T = true_head_num

    def rel(head_slice: slice) -> float:
        base = edge_g0[head_slice]
        delta = diff[head_slice]

        if base.numel() == 0:
            return 0.0

        return (delta.norm() / (base.norm() + eps)).item()

    return {
        "all": rel(slice(0, H)),
        "true": rel(slice(0, T)),
        "proxy": rel(slice(T, H)),
    }


def summarize(values: List[float]) -> Tuple[float, float, float]:
    if len(values) == 0:
        return 0.0, 0.0, 0.0

    arr = np.asarray(values, dtype=np.float64)

    return float(arr.mean()), float(np.median(arr)), float(arr.max())


# ============================================================
# Main experiment
# ============================================================

def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = torch.device(args.device)

    model = build_zerosim_model(args).to(device)
    load_checkpoint_if_needed(model, args.checkpoint)

    model.eval()

    x = torch.randn(
        1,
        args.device_num,
        args.feature_dim,
        device=device,
    )

    hidden, parameter_tensors, target_encoder, attn_mask = get_zerosim_hidden_before_target_encoder(
        model=model,
        x=x,
        parameter_layer_idx=args.parameter_layer_idx,
        target_encoder_name=args.target_encoder,
    )

    hidden = hidden.to(device)
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)

    if attn_mask is None:
        raise RuntimeError(
            "Selected target_encoder has attn_mask=None. "
            "For two-hop topology coupling, use --target_encoder structure_refining."
        )

    adj = additive_mask_to_adj(
        attn_mask=attn_mask.detach(),
        exclude_global=True,
    )

    triplets = find_two_hop_triplets(
        adj=adj,
        max_triplets=args.samples,
        seed=args.seed,
    )

    if len(triplets) == 0:
        raise RuntimeError(
            "No valid two-hop triplets found. "
            "Do not use --mask full. Try --mask local or --mask random."
        )

    true_head_num = int(round(args.num_heads * args.true_head_ratio))
    true_head_num = max(0, min(args.num_heads, true_head_num))

    print("=" * 100)
    print("Two-hop gradient coupling experiment on ZeroSim Encoder attention")
    print("=" * 100)
    print(f"model_name              : {args.model_name}")
    print(f"checkpoint              : {args.checkpoint}")
    print(f"mask                    : {args.mask}")
    print(f"device_num              : {args.device_num}")
    print(f"hidden_dim              : {args.hidden_dim}")
    print(f"num_heads               : {args.num_heads}")
    print(f"true_head_ratio         : {args.true_head_ratio}")
    print(f"true/proxy heads        : {true_head_num}/{args.num_heads - true_head_num}")
    print(f"parameter_layer_idx     : {args.parameter_layer_idx}")
    print(f"target_encoder          : {args.target_encoder}")
    print(f"hidden shape            : {tuple(hidden.shape)}")
    print(f"attn_mask shape         : {tuple(attn_mask.shape)}")
    print(f"valid triplets          : {len(triplets)}")
    print(f"perturb_eps             : {args.perturb_eps}")
    print("=" * 100)

    modes = ["full", "latest_custom"]

    results = []

    for mode in modes:
        group_values = {
            "all": [],
            "true": [],
            "proxy": [],
        }

        # Use train mode only for enabling normal autograd behavior.
        # Dropout is manually not applied in the probe.
        target_encoder.train()

        for idx, triplet in enumerate(triplets):
            values = measure_one_triplet(
                encoder=target_encoder,
                hidden=hidden,
                attn_mask=attn_mask,
                triplet=triplet,
                mode=mode,
                true_head_num=true_head_num,
                perturb_eps=args.perturb_eps,
                seed=args.seed + 1000 + idx,
            )

            for group, val in values.items():
                group_values[group].append(val)

        print(f"\nMode: {mode}")
        print("-" * 100)

        for group in ["all", "true", "proxy"]:
            mean_v, median_v, max_v = summarize(group_values[group])

            print(
                f"{group:>8s} | "
                f"mean={mean_v:.6e} | "
                f"median={median_v:.6e} | "
                f"max={max_v:.6e} | "
                f"n={len(group_values[group])}"
            )

            results.append({
                "mode": mode,
                "group": group,
                "mean_rel_change": mean_v,
                "median_rel_change": median_v,
                "max_rel_change": max_v,
                "sample_num": len(group_values[group]),
            })

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "group",
                "mean_rel_change",
                "median_rel_change",
                "max_rel_change",
                "sample_num",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 100)
    print(f"Saved CSV to: {args.output_csv}")
    print("=" * 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--model_name",
        type=str,
        default="zerosim_device",
        choices=[
            "zerosim_device",
            "zerosim_device_final_no_grad_test",
            "zerosim_device_no_grad_test",
        ],
    )

    parser.add_argument("--checkpoint", type=str, default=None)

    parser.add_argument("--device_num", type=int, default=32)
    parser.add_argument("--feature_dim", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num_heads", type=int, default=8)

    parser.add_argument("--embedding_layer_num", type=int, default=2)
    parser.add_argument("--encoder_layer_num", type=int, default=3)
    parser.add_argument("--decoder_layer_num", type=int, default=1)
    parser.add_argument("--output_layer_num", type=int, default=2)
    parser.add_argument("--performance_num", type=int, default=5)

    parser.add_argument("--structure_encoding_layer_num", type=int, default=3)
    parser.add_argument("--parameter_injection_layer_num", type=int, default=3)

    parser.add_argument("--gcn_layer_num", type=int, default=3)
    parser.add_argument("--gat_layer_num", type=int, default=3)

    parser.add_argument("--init_alpha", type=float, default=1.0)
    parser.add_argument("--true_head_ratio", type=float, default=0.5)
    parser.add_argument("--proxy_scale", type=float, default=1.0)
    parser.add_argument("--freeze_proxy_attention", action="store_true")

    parser.add_argument(
        "--mask",
        type=str,
        default="local",
        choices=["local", "random", "full"],
        help="Use local or random. Full mask has no non-adjacent co-neighbor triplets.",
    )

    parser.add_argument(
        "--parameter_layer_idx",
        type=int,
        default=0,
        help="Which ZeroSim parameter_injection_layer to probe.",
    )

    parser.add_argument(
        "--target_encoder",
        type=str,
        default="structure_refining",
        choices=["structure_refining", "context_enhancing"],
        help="Use structure_refining for topology-masked attention.",
    )

    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--perturb_eps", type=float, default=1e-3)

    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/two_hop_gradient_coupling_zerosim.csv",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[Warning] CUDA not available. Use CPU instead of {args.device}.")
        args.device = "cpu"

    run(args)


if __name__ == "__main__":
    main()