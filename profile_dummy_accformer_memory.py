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
        "embedding_layer_num": args.embedding_layer_num,
        "encoder_layer_num": args.encoder_layer_num,
        "decoder_layer_num": args.decoder_layer_num,
        "output_layer_num": args.output_layer_num,
        "performance_num": args.performance_num,
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
    parser.add_argument("--embedding_layer_num", type=int, default=2)
    parser.add_argument("--encoder_layer_num", type=int, default=3)
    parser.add_argument("--decoder_layer_num", type=int, default=1)
    parser.add_argument("--output_layer_num", type=int, default=2)
    parser.add_argument("--performance_num", type=int, default=5)

    parser.add_argument("--mask", type=str, default="full", choices=["full", "local", "random"])
    parser.add_argument("--repeat", type=int, default=20)

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
    print(f"batch_size       = {args.batch_size}")
    print(f"device_num       = {args.device_num}")
    print(f"feature_dim      = {args.feature_dim}")
    print(f"hidden_dim       = {args.hidden_dim}")
    print(f"encoder_layers   = {args.encoder_layer_num}")
    print(f"num_heads        = {args.num_heads}")
    print(f"mask             = {args.mask}")
    print(f"profile_graph    = {args.profile_graph}")
    print("=" * 100)

    results = {}
    graph_results = {}

    for model_name in ["accformer", "accformer_no_grad"]:
        print(f"\nProfiling {model_name} ...")

        reset_memory()
        model = build_one_model(model_name, args, device)

        param_num = sum(p.numel() for p in model.parameters())
        print(f"Params: {param_num:,}")

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

    base = results["accformer"]
    test = results["accformer_no_grad"]

    infer_mem_diff = test["infer_peak"] - base["infer_peak"]
    train_mem_diff = test["train_peak"] - base["train_peak"]

    infer_time_diff = test["infer_time"] - base["infer_time"]
    train_time_diff = test["train_time"] - base["train_time"]

    print(f"Inference memory accformer        : {base['infer_peak']:.2f} MB")
    print(f"Inference memory accformer_no_grad: {test['infer_peak']:.2f} MB")
    print(f"Inference memory diff             : {infer_mem_diff:.2f} MB")

    print()

    print(f"Training memory accformer         : {base['train_peak']:.2f} MB")
    print(f"Training memory accformer_no_grad : {test['train_peak']:.2f} MB")
    print(f"Training memory diff              : {train_mem_diff:.2f} MB")

    print()

    print(f"Inference time accformer          : {base['infer_time'] * 1000:.2f} ms")
    print(f"Inference time accformer_no_grad  : {test['infer_time'] * 1000:.2f} ms")
    print(f"Inference time diff               : {infer_time_diff * 1000:.2f} ms")

    print()

    print(f"Training time accformer           : {base['train_time'] * 1000:.2f} ms")
    print(f"Training time accformer_no_grad   : {test['train_time'] * 1000:.2f} ms")
    print(f"Training time diff                : {train_time_diff * 1000:.2f} ms")

    if args.profile_graph:
        print("\n" + "=" * 100)
        print("Graph Summary: accformer_no_grad - accformer")
        print("=" * 100)

        g_base = graph_results["accformer"]
        g_test = graph_results["accformer_no_grad"]

        keys = [
            "graph_nodes",
            "num_saved",
            "total_saved_mb",
            "unique_storage_mb",
            "forward_peak_mb",
            "train_peak_mb",
            "elapsed_ms",
        ]

        for key in keys:
            base_value = g_base[key]
            test_value = g_test[key]
            diff_value = test_value - base_value
            rel_value = diff_value / base_value * 100.0 if base_value != 0 else 0.0

            print(
                f"{key:20s}: "
                f"accformer={base_value:10.2f} | "
                f"no_grad={test_value:10.2f} | "
                f"diff={diff_value:10.2f} | "
                f"rel={rel_value:+8.2f}%"
            )


if __name__ == "__main__":
    main()