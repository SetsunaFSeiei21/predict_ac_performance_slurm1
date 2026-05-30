import argparse
import gc
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.autograd.graph import saved_tensors_hooks

from src.base import build_model


def cuda_sync():
    torch.cuda.synchronize()


def reset_memory():
    gc.collect()
    torch.cuda.empty_cache()
    cuda_sync()
    torch.cuda.reset_peak_memory_stats()


def get_peak_mb():
    cuda_sync()
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def make_dummy_device_messages(device_num: int):
    kinds = ["NMOS", "PMOS", "RES", "CAP", "V"]
    return [{"kind": kinds[i % len(kinds)]} for i in range(device_num)]


def make_attn_mask(device_num: int, mode: str = "full"):
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


def build_one_model(model_name, args, device):
    device_messages = make_dummy_device_messages(args.device_num)
    attn_mask = make_attn_mask(args.device_num, args.mask)

    input_shape = (args.device_num, args.feature_dim)

    model_hparams = {
        "hidden_dim": args.hidden_dim,
        "output_dim": args.output_dim,
        "dropout": args.dropout,
        "num_heads": args.num_heads,

        # Common layers
        "embedding_layer_num": args.embedding_layer_num,
        "encoder_layer_num": args.encoder_layer_num,
        "decoder_layer_num": args.decoder_layer_num,
        "output_layer_num": args.output_layer_num,
        "performance_num": args.performance_num,

        # ZeroSim-specific layers
        "structure_encoding_layer_num": args.structure_encoding_layer_num,
        "parameter_injection_layer_num": args.parameter_injection_layer_num,

        # GCN/GAT-specific layers
        "gcn_layer_num": args.gcn_layer_num,
        "gat_layer_num": args.gat_layer_num,

        # No-grad / proxy-gradient attention parameters
        "init_alpha": args.init_alpha,
        "true_head_ratio": args.true_head_ratio,
        "need_attn_score": args.need_attn_score,
        "proxy_scale": args.proxy_scale,
        "freeze_proxy_attention": args.freeze_proxy_attention,
    }

    model = build_model(
        model_name=model_name,
        model_hyper_parameters=model_hparams,
        input_shape=input_shape,
        device_messages=device_messages,
        device_level_attn_mask=attn_mask,
    )

    return model.to(device)


@torch.no_grad()
def profile_inference(model, x, repeat):
    model.eval()

    for _ in range(3):
        _ = model(x)
    cuda_sync()

    reset_memory()
    start = time.time()

    for _ in range(repeat):
        _ = model(x)

    cuda_sync()
    elapsed = (time.time() - start) / repeat
    peak_mb = get_peak_mb()

    return peak_mb, elapsed


def profile_training(model, x, y, repeat):
    model.train()
    loss_fn = nn.MSELoss()

    for _ in range(3):
        model.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
    cuda_sync()

    reset_memory()
    start = time.time()

    for _ in range(repeat):
        model.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()

    cuda_sync()
    elapsed = (time.time() - start) / repeat
    peak_mb = get_peak_mb()

    return peak_mb, elapsed


def count_autograd_nodes(loss: torch.Tensor) -> int:
    if loss.grad_fn is None:
        return 0

    seen = set()
    stack = [loss.grad_fn]

    while stack:
        fn = stack.pop()
        if fn is None or fn in seen:
            continue

        seen.add(fn)

        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                stack.append(next_fn)

    return len(seen)


class SavedTensorStats:
    def __init__(self):
        self.num_saved = 0
        self.total_saved_bytes = 0
        self.unique_storage_bytes = 0
        self.seen_storages = set()
        self.shape_counter = Counter()

    def pack(self, tensor: torch.Tensor):
        self.num_saved += 1

        tensor_bytes = tensor.numel() * tensor.element_size()
        self.total_saved_bytes += tensor_bytes

        try:
            storage = tensor.untyped_storage()
            storage_key = (
                str(tensor.device),
                storage.data_ptr(),
                storage.nbytes(),
            )
            if storage_key not in self.seen_storages:
                self.seen_storages.add(storage_key)
                self.unique_storage_bytes += storage.nbytes()
        except Exception:
            pass

        shape_key = (
            tuple(tensor.shape),
            str(tensor.dtype),
            str(tensor.device),
            bool(tensor.requires_grad),
        )
        self.shape_counter[shape_key] += 1

        return tensor

    def unpack(self, tensor: torch.Tensor):
        return tensor

    def summary(self):
        mb = 1024 * 1024
        return {
            "num_saved": self.num_saved,
            "total_saved_mb": self.total_saved_bytes / mb,
            "unique_storage_mb": self.unique_storage_bytes / mb,
        }

    def top_shapes_text(self, topk: int = 10):
        lines = []
        for (shape, dtype, device, requires_grad), count in self.shape_counter.most_common(topk):
            lines.append(
                f"    count={count:4d} | shape={shape} | dtype={dtype} | "
                f"device={device} | requires_grad={requires_grad}"
            )
        return "\n".join(lines)


def profile_graph(model, x, y, topk=10):
    model.train()
    loss_fn = nn.MSELoss()

    for _ in range(3):
        model.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
    cuda_sync()

    reset_memory()

    stats = SavedTensorStats()
    model.zero_grad(set_to_none=True)

    start = time.time()

    with saved_tensors_hooks(stats.pack, stats.unpack):
        pred = model(x)
        loss = loss_fn(pred, y)

    graph_nodes = count_autograd_nodes(loss)
    forward_peak_mb = get_peak_mb()

    loss.backward()
    cuda_sync()

    train_peak_mb = get_peak_mb()
    elapsed_ms = (time.time() - start) * 1000.0

    result = stats.summary()
    result.update(
        {
            "graph_nodes": graph_nodes,
            "forward_peak_mb": forward_peak_mb,
            "train_peak_mb": train_peak_mb,
            "elapsed_ms": elapsed_ms,
            "loss": float(loss.detach().cpu().item()),
            "top_shapes": stats.top_shapes_text(topk),
        }
    )

    return result


def print_memory_result(model_name, infer_peak, infer_time, train_peak, train_time):
    print(f"Inference peak memory: {infer_peak:.2f} MB")
    print(f"Inference time       : {infer_time * 1000:.2f} ms/iter")
    print(f"Training peak memory : {train_peak:.2f} MB")
    print(f"Training time        : {train_time * 1000:.2f} ms/iter")


def print_graph_result(model_name, graph_result):
    print(f"\nGraph profile for {model_name}:")
    print(f"Autograd nodes      : {graph_result['graph_nodes']}")
    print(f"Saved tensors       : {graph_result['num_saved']}")
    print(f"Total saved tensors : {graph_result['total_saved_mb']:.2f} MB")
    print(f"Unique storage      : {graph_result['unique_storage_mb']:.2f} MB")
    print(f"Forward peak memory : {graph_result['forward_peak_mb']:.2f} MB")
    print(f"Train peak memory   : {graph_result['train_peak_mb']:.2f} MB")
    print(f"One-step time       : {graph_result['elapsed_ms']:.2f} ms")
    print(f"Loss                : {graph_result['loss']:.6f}")
    print("Top saved tensor shapes:")
    print(graph_result["top_shapes"])


def print_relative_result(value, base_value):
    diff = value - base_value
    rel = diff / base_value * 100.0 if base_value != 0 else 0.0
    return diff, rel


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device_num", type=int, default=32)
    parser.add_argument("--feature_dim", type=int, default=8)

    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_heads", type=int, default=8)

    # Common architecture parameters
    parser.add_argument("--embedding_layer_num", type=int, default=2)
    parser.add_argument("--encoder_layer_num", type=int, default=3)
    parser.add_argument("--decoder_layer_num", type=int, default=1)
    parser.add_argument("--output_layer_num", type=int, default=2)
    parser.add_argument("--performance_num", type=int, default=5)

    # ZeroSim-specific architecture parameters
    parser.add_argument(
        "--structure_encoding_layer_num",
        type=int,
        default=3,
        help="Number of structure encoding layers for ZeroSim-style models.",
    )
    parser.add_argument(
        "--parameter_injection_layer_num",
        type=int,
        default=3,
        help="Number of parameter injection layers for ZeroSim-style models.",
    )

    # GCN/GAT-specific architecture parameters
    parser.add_argument(
        "--gcn_layer_num",
        type=int,
        default=3,
        help="Number of GCN layers for GCN-style models.",
    )
    parser.add_argument(
        "--gat_layer_num",
        type=int,
        default=3,
        help="Number of GAT layers for GAT-style models.",
    )

    # No-grad / proxy-gradient attention parameters
    parser.add_argument(
        "--init_alpha",
        type=float,
        default=1.0,
        help="Initial alpha for ACCFormer/ZeroSim no-grad attention models.",
    )
    parser.add_argument(
        "--true_head_ratio",
        type=float,
        default=0.5,
        help="Ratio of heads that keep true score-gradient.",
    )
    parser.add_argument(
        "--need_attn_score",
        action="store_true",
        help="Whether to explicitly return/store attention scores for supported models.",
    )
    parser.add_argument(
        "--proxy_scale",
        type=float,
        default=1.0,
        help="Scale of value-gradient proxy injected into score projection.",
    )
    parser.add_argument(
        "--freeze_proxy_attention",
        action="store_true",
        help="Detach attention vectors of proxy heads for supported GAT no-grad models.",
    )

    parser.add_argument("--mask", type=str, default="full", choices=["full", "local", "random"])
    parser.add_argument("--repeat", type=int, default=20)

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["gat", "gat_split_full", "gat_no_grad_test"],
        help=(
            "Models to profile. Example: "
            "--models gat gat_no_grad_test"
        ),
    )

    parser.add_argument(
        "--profile_graph",
        action="store_true",
        help="Also profile autograd graph nodes and saved tensors.",
    )
    parser.add_argument("--topk", type=int, default=10)

    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is not available."

    device = torch.device(args.device)
    torch.manual_seed(42)
    np.random.seed(42)

    x = torch.randn(
        args.batch_size,
        args.device_num,
        args.feature_dim,
        device=device,
    )

    y = torch.randn(
        args.batch_size,
        args.performance_num * args.output_dim,
        device=device,
    )

    print("=" * 100)
    print(f"models                         = {args.models}")
    print(f"batch_size                     = {args.batch_size}")
    print(f"device_num                     = {args.device_num}")
    print(f"feature_dim                    = {args.feature_dim}")
    print(f"hidden_dim                     = {args.hidden_dim}")
    print(f"output_dim                     = {args.output_dim}")
    print(f"dropout                        = {args.dropout}")
    print(f"num_heads                      = {args.num_heads}")
    print(f"embedding_layer_num            = {args.embedding_layer_num}")
    print(f"encoder_layer_num              = {args.encoder_layer_num}")
    print(f"structure_encoding_layer_num   = {args.structure_encoding_layer_num}")
    print(f"parameter_injection_layer_num  = {args.parameter_injection_layer_num}")
    print(f"gcn_layer_num                  = {args.gcn_layer_num}")
    print(f"gat_layer_num                  = {args.gat_layer_num}")
    print(f"decoder_layer_num              = {args.decoder_layer_num}")
    print(f"output_layer_num               = {args.output_layer_num}")
    print(f"performance_num                = {args.performance_num}")
    print(f"init_alpha                     = {args.init_alpha}")
    print(f"true_head_ratio                = {args.true_head_ratio}")
    print(f"need_attn_score                = {args.need_attn_score}")
    print(f"proxy_scale                    = {args.proxy_scale}")
    print(f"freeze_proxy_attention         = {args.freeze_proxy_attention}")
    print(f"mask                           = {args.mask}")
    print(f"repeat                         = {args.repeat}")
    print(f"profile_graph                  = {args.profile_graph}")
    print("=" * 100)

    results = {}
    graph_results = {}

    for model_name in args.models:
        print(f"\nProfiling {model_name} ...")

        reset_memory()
        model = build_one_model(model_name, args, device)

        param_num = sum(p.numel() for p in model.parameters())
        trainable_param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Params          : {param_num:,}")
        print(f"Trainable params: {trainable_param_num:,}")

        infer_peak, infer_time = profile_inference(model, x, args.repeat)

        del model
        reset_memory()

        model = build_one_model(model_name, args, device)
        train_peak, train_time = profile_training(model, x, y, args.repeat)

        results[model_name] = {
            "infer_peak": infer_peak,
            "infer_time": infer_time,
            "train_peak": train_peak,
            "train_time": train_time,
            "param_num": param_num,
            "trainable_param_num": trainable_param_num,
        }

        print_memory_result(model_name, infer_peak, infer_time, train_peak, train_time)

        del model
        reset_memory()

        if args.profile_graph:
            model = build_one_model(model_name, args, device)
            graph_result = profile_graph(model, x, y, topk=args.topk)
            graph_results[model_name] = graph_result
            print_graph_result(model_name, graph_result)

            del model
            reset_memory()

    print("\n" + "=" * 100)
    print("Summary")
    print("=" * 100)

    base_name = args.models[0]
    base = results[base_name]

    for model_name in args.models:
        cur = results[model_name]

        infer_mem_diff, infer_mem_rel = print_relative_result(cur["infer_peak"], base["infer_peak"])
        train_mem_diff, train_mem_rel = print_relative_result(cur["train_peak"], base["train_peak"])
        infer_time_diff, infer_time_rel = print_relative_result(cur["infer_time"], base["infer_time"])
        train_time_diff, train_time_rel = print_relative_result(cur["train_time"], base["train_time"])
        param_diff, param_rel = print_relative_result(cur["param_num"], base["param_num"])
        trainable_param_diff, trainable_param_rel = print_relative_result(
            cur["trainable_param_num"],
            base["trainable_param_num"],
        )

        print(f"\nModel: {model_name}")
        print(
            f"Params              : {cur['param_num']:12,.0f} | "
            f"diff vs {base_name}: {param_diff:+12,.0f} ({param_rel:+.2f}%)"
        )
        print(
            f"Trainable params    : {cur['trainable_param_num']:12,.0f} | "
            f"diff vs {base_name}: {trainable_param_diff:+12,.0f} ({trainable_param_rel:+.2f}%)"
        )
        print(
            f"Inference memory    : {cur['infer_peak']:12.2f} MB | "
            f"diff vs {base_name}: {infer_mem_diff:+12.2f} MB ({infer_mem_rel:+.2f}%)"
        )
        print(
            f"Training memory     : {cur['train_peak']:12.2f} MB | "
            f"diff vs {base_name}: {train_mem_diff:+12.2f} MB ({train_mem_rel:+.2f}%)"
        )
        print(
            f"Inference time      : {cur['infer_time'] * 1000:12.2f} ms | "
            f"diff vs {base_name}: {infer_time_diff * 1000:+12.2f} ms ({infer_time_rel:+.2f}%)"
        )
        print(
            f"Training time       : {cur['train_time'] * 1000:12.2f} ms | "
            f"diff vs {base_name}: {train_time_diff * 1000:+12.2f} ms ({train_time_rel:+.2f}%)"
        )

    if args.profile_graph:
        print("\n" + "=" * 100)
        print("Graph Summary")
        print("=" * 100)

        g_base = graph_results[base_name]

        keys = [
            "graph_nodes",
            "num_saved",
            "total_saved_mb",
            "unique_storage_mb",
            "forward_peak_mb",
            "train_peak_mb",
            "elapsed_ms",
        ]

        for model_name in args.models:
            g_cur = graph_results[model_name]

            print(f"\nModel: {model_name}")

            for key in keys:
                base_value = g_base[key]
                cur_value = g_cur[key]
                diff_value = cur_value - base_value
                rel_value = diff_value / base_value * 100.0 if base_value != 0 else 0.0

                print(
                    f"{key:20s}: "
                    f"{cur_value:12.2f} | "
                    f"diff vs {base_name}: {diff_value:+12.2f} | "
                    f"rel={rel_value:+8.2f}%"
                )


if __name__ == "__main__":
    main()