import itertools
import json
import math
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm

from src.base import build_model
from src.base.data_engine import build_full_train_dataloader
from src.dataset.dataloader import recover_y_to_original_scale


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def expand_hyper_parameters(cfg) -> List[Dict[str, Any]]:
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    exp_cfg = OmegaConf.to_container(cfg.exp, resolve=True)

    model_name = model_cfg.pop("name", None)

    if model_name is None:
        model_name = getattr(cfg, "model_name", None)

    if model_name is None:
        raise ValueError("Model name is missing. Please set cfg.model.name.")

    search_space = {}
    search_space.update(model_cfg)

    for key in ["lr", "weight_decay", "batch_size"]:
        if key in exp_cfg:
            search_space[key] = exp_cfg[key]

    keys = []
    values = []

    for key, value in search_space.items():
        keys.append(key)

        if isinstance(value, list):
            values.append(value)
        else:
            values.append([value])

    all_hypers = []

    for item in itertools.product(*values):
        hyper = dict(zip(keys, item))
        hyper["name"] = model_name
        all_hypers.append(hyper)

    return all_hypers


def build_optimizer(cfg, model: nn.Module, hyper: Dict[str, Any]):
    lr = float(hyper.get("lr", cfg.exp.lr))
    weight_decay = float(hyper.get("weight_decay", cfg.exp.weight_decay))

    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )


def build_scheduler(cfg, optimizer):
    scheduler_cfg = cfg.scheduler
    scheduler_name = scheduler_cfg.name.lower()

    if scheduler_name == "none":
        return None

    if scheduler_name == "warmup_cosine":
        total_epochs = int(cfg.exp.epochs)
        warmup_epochs = int(scheduler_cfg.warmup_epochs)
        warmup_start_factor = float(scheduler_cfg.warmup_start_factor)
        min_lr_factor = float(scheduler_cfg.min_lr_factor)

        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0")

        if warmup_epochs >= total_epochs:
            raise ValueError(
                f"warmup_epochs={warmup_epochs} must be smaller than "
                f"total_epochs={total_epochs}"
            )

        def lr_lambda(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                progress = epoch / max(1, warmup_epochs)
                return warmup_start_factor + progress * (1.0 - warmup_start_factor)

            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

            return min_lr_factor + (1.0 - min_lr_factor) * cosine_factor

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda,
        )

    raise ValueError(f"Unsupported scheduler name: {scheduler_cfg.name}")


def calc_r2_np(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    y_true_mean = np.mean(y_true, axis=0, keepdims=True)

    ss_res = np.sum((y_pred - y_true) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true_mean) ** 2, axis=0)

    eps = 1e-12
    return 1.0 - ss_res / (ss_tot + eps)


def predict_loader(
    model: nn.Module,
    data_loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    pred_lst = []
    true_lst = []

    with torch.no_grad():
        for batch in data_loader:
            x = batch["device_features"].to(device)
            y = batch["targets"].to(device)

            pred = model(x)

            pred_lst.append(pred.detach().cpu().numpy())
            true_lst.append(y.detach().cpu().numpy())

    pred_np = np.concatenate(pred_lst, axis=0)
    true_np = np.concatenate(true_lst, axis=0)

    return pred_np, true_np


def run_one_epoch(
    model: nn.Module,
    data_loader,
    optimizer,
    device: torch.device,
    alpha: float,
    train: bool,
) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    mse_fn = nn.MSELoss(reduction="mean")
    mae_fn = nn.L1Loss(reduction="mean")

    total_loss_sum = 0.0
    mse_loss_sum = 0.0
    mae_loss_sum = 0.0
    sample_num = 0

    for batch in data_loader:
        x = batch["device_features"].to(device)
        y = batch["targets"].to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            pred = model(x)

            if pred.shape != y.shape:
                raise ValueError(
                    f"Prediction shape {pred.shape} != target shape {y.shape}"
                )

            mse_loss = mse_fn(pred, y)
            mae_loss = mae_fn(pred, y)
            loss = alpha * mae_loss + (1.0 - alpha) * mse_loss

            if train:
                loss.backward()
                optimizer.step()

        batch_size = x.shape[0]

        total_loss_sum += float(loss.item()) * batch_size
        mse_loss_sum += float(mse_loss.item()) * batch_size
        mae_loss_sum += float(mae_loss.item()) * batch_size
        sample_num += batch_size

    return {
        "loss": total_loss_sum / sample_num,
        "mse_loss": mse_loss_sum / sample_num,
        "mae_loss": mae_loss_sum / sample_num,
    }


def evaluate_r2(
    model: nn.Module,
    data_loader,
    y_scaler,
    log_y_indices: List[int],
    device: torch.device,
) -> Dict[str, Any]:
    pred_norm, true_norm = predict_loader(
        model=model,
        data_loader=data_loader,
        device=device,
    )

    pred_raw = recover_y_to_original_scale(
        y=pred_norm,
        y_scaler=y_scaler,
        log_y_indices=log_y_indices,
    )

    true_raw = recover_y_to_original_scale(
        y=true_norm,
        y_scaler=y_scaler,
        log_y_indices=log_y_indices,
    )

    r2_per_metric = calc_r2_np(pred_raw, true_raw)

    return {
        "r2_per_metric": r2_per_metric.tolist(),
        "mean_r2": float(np.mean(r2_per_metric)),
    }


def train_one_fold(
    cfg,
    hyper: Dict[str, Any],
    fold_idx: int,
    fold_loader_dict: Dict[str, Any],
    input_shape: tuple[int, int],
    device_messages: List[Dict[str, Any]],
    device_level_attn_mask: np.ndarray,
    log_y_indices: List[int],
    device: torch.device,
    logger,
) -> Dict[str, Any]:
    model = build_model(
        model_name=hyper["name"],
        model_hyper_parameters=hyper,
        input_shape=input_shape,
        device_messages=device_messages,
        device_level_attn_mask=device_level_attn_mask,
    ).to(device)

    optimizer = build_optimizer(cfg, model, hyper)
    scheduler = build_scheduler(cfg, optimizer)

    train_loader = fold_loader_dict["train_loader"]
    valid_loader = fold_loader_dict["valid_loader"]

    train_loss_curve = []
    train_mse_curve = []
    train_mae_curve = []

    val_loss_curve = []
    val_mse_curve = []
    val_mae_curve = []

    epochs = int(cfg.exp.epochs)
    alpha = float(cfg.exp.alpha)
    log_interval = int(cfg.exp.log_interval)

    progress = tqdm(
        range(epochs),
        desc=f"Fold {fold_idx + 1}/{cfg.exp.k_fold}",
        leave=False,
    )

    for epoch in progress:
        train_metrics = run_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            alpha=alpha,
            train=True,
        )

        val_metrics = run_one_epoch(
            model=model,
            data_loader=valid_loader,
            optimizer=None,
            device=device,
            alpha=alpha,
            train=False,
        )

        if scheduler is not None:
            scheduler.step()

        train_loss_curve.append(train_metrics["loss"])
        train_mse_curve.append(train_metrics["mse_loss"])
        train_mae_curve.append(train_metrics["mae_loss"])

        val_loss_curve.append(val_metrics["loss"])
        val_mse_curve.append(val_metrics["mse_loss"])
        val_mae_curve.append(val_metrics["mae_loss"])

        progress.set_postfix(
            {
                "train_loss": f"{train_metrics['loss']:.4f}",
                "val_loss": f"{val_metrics['loss']:.4f}",
            }
        )

        if (epoch + 1) % log_interval == 0 or (epoch + 1) == epochs:
            logger.info(
                f"Fold {fold_idx + 1} | "
                f"Epoch {epoch + 1}/{epochs} | "
                f"train_loss={train_metrics['loss']:.6f} | "
                f"val_loss={val_metrics['loss']:.6f}"
            )

    train_r2 = evaluate_r2(
        model=model,
        data_loader=train_loader,
        y_scaler=fold_loader_dict["y_scaler"],
        log_y_indices=log_y_indices,
        device=device,
    )

    val_r2 = evaluate_r2(
        model=model,
        data_loader=valid_loader,
        y_scaler=fold_loader_dict["y_scaler"],
        log_y_indices=log_y_indices,
        device=device,
    )

    del model, optimizer, scheduler

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "train_loss": train_loss_curve,
        "train_mse_loss": train_mse_curve,
        "train_mae_loss": train_mae_curve,

        "val_loss": val_loss_curve,
        "val_mse_loss": val_mse_curve,
        "val_mae_loss": val_mae_curve,

        "train_r2_per_metric": train_r2["r2_per_metric"],
        "train_mean_r2": train_r2["mean_r2"],

        "val_r2_per_metric": val_r2["r2_per_metric"],
        "val_mean_r2": val_r2["mean_r2"],
    }


def mean_curve(curves: List[List[float]]) -> List[float]:
    return np.asarray(curves, dtype=np.float64).mean(axis=0).tolist()

def _stat_ddof(num_items: int) -> int:
    return 1 if num_items > 1 else 0


def std_curve(curves: List[List[float]]) -> List[float]:
    arr = np.asarray(curves, dtype=np.float64)
    return arr.std(axis=0, ddof=_stat_ddof(arr.shape[0])).tolist()


def var_curve(curves: List[List[float]]) -> List[float]:
    arr = np.asarray(curves, dtype=np.float64)
    return arr.var(axis=0, ddof=_stat_ddof(arr.shape[0])).tolist()

def summarize_fold_results(
    hyper: Dict[str, Any],
    fold_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    all_train_loss = [item["train_loss"] for item in fold_results]
    all_train_mse = [item["train_mse_loss"] for item in fold_results]
    all_train_mae = [item["train_mae_loss"] for item in fold_results]

    all_val_loss = [item["val_loss"] for item in fold_results]
    all_val_mse = [item["val_mse_loss"] for item in fold_results]
    all_val_mae = [item["val_mae_loss"] for item in fold_results]

    all_train_r2_per_metric = [item["train_r2_per_metric"] for item in fold_results]
    all_val_r2_per_metric = [item["val_r2_per_metric"] for item in fold_results]

    all_train_mean_r2 = [item["train_mean_r2"] for item in fold_results]
    all_val_mean_r2 = [item["val_mean_r2"] for item in fold_results]

    train_r2_np = np.asarray(all_train_r2_per_metric, dtype=np.float64)
    val_r2_np = np.asarray(all_val_r2_per_metric, dtype=np.float64)

    train_mean_r2_np = np.asarray(all_train_mean_r2, dtype=np.float64)
    val_mean_r2_np = np.asarray(all_val_mean_r2, dtype=np.float64)

    ddof = _stat_ddof(len(fold_results))

    return {
        "hyper_parameters": deepcopy(hyper),
        "num_folds": len(fold_results),

        "all_train_loss": all_train_loss,
        "all_train_mse_loss": all_train_mse,
        "all_train_mae_loss": all_train_mae,

        "all_val_loss": all_val_loss,
        "all_val_mse_loss": all_val_mse,
        "all_val_mae_loss": all_val_mae,

        "all_train_r2_per_metric": all_train_r2_per_metric,
        "all_val_r2_per_metric": all_val_r2_per_metric,

        "all_train_mean_r2": all_train_mean_r2,
        "all_val_mean_r2": all_val_mean_r2,

        "mean_train_loss": mean_curve(all_train_loss),
        "mean_train_mse_loss": mean_curve(all_train_mse),
        "mean_train_mae_loss": mean_curve(all_train_mae),

        "mean_val_loss": mean_curve(all_val_loss),
        "mean_val_mse_loss": mean_curve(all_val_mse),
        "mean_val_mae_loss": mean_curve(all_val_mae),

        "std_train_loss": std_curve(all_train_loss),
        "std_train_mse_loss": std_curve(all_train_mse),
        "std_train_mae_loss": std_curve(all_train_mae),

        "std_val_loss": std_curve(all_val_loss),
        "std_val_mse_loss": std_curve(all_val_mse),
        "std_val_mae_loss": std_curve(all_val_mae),

        "var_train_loss": var_curve(all_train_loss),
        "var_train_mse_loss": var_curve(all_train_mse),
        "var_train_mae_loss": var_curve(all_train_mae),

        "var_val_loss": var_curve(all_val_loss),
        "var_val_mse_loss": var_curve(all_val_mse),
        "var_val_mae_loss": var_curve(all_val_mae),

        "mean_train_r2_per_metric": train_r2_np.mean(axis=0).tolist(),
        "mean_val_r2_per_metric": val_r2_np.mean(axis=0).tolist(),

        "std_train_r2_per_metric": train_r2_np.std(axis=0, ddof=ddof).tolist(),
        "std_val_r2_per_metric": val_r2_np.std(axis=0, ddof=ddof).tolist(),

        "var_train_r2_per_metric": train_r2_np.var(axis=0, ddof=ddof).tolist(),
        "var_val_r2_per_metric": val_r2_np.var(axis=0, ddof=ddof).tolist(),

        "mean_train_r2": float(train_mean_r2_np.mean()),
        "mean_val_r2": float(val_mean_r2_np.mean()),

        "std_train_r2": float(train_mean_r2_np.std(ddof=ddof)),
        "std_val_r2": float(val_mean_r2_np.std(ddof=ddof)),

        "var_train_r2": float(train_mean_r2_np.var(ddof=ddof)),
        "var_val_r2": float(val_mean_r2_np.var(ddof=ddof)),
    }


def dumps_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def save_summaries_to_excel(
    summaries: List[Dict[str, Any]],
    save_path: str,
) -> None:
    rows = []

    for hyper_idx, summary in enumerate(summaries):
        row = {
            "hyper_id": hyper_idx,
            "hyper_parameters": dumps_json(summary["hyper_parameters"]),

            "mean_train_r2": summary["mean_train_r2"],
            "mean_val_r2": summary["mean_val_r2"],

            "mean_train_r2_per_metric": dumps_json(summary["mean_train_r2_per_metric"]),
            "mean_val_r2_per_metric": dumps_json(summary["mean_val_r2_per_metric"]),

            "all_train_r2_per_metric": dumps_json(summary["all_train_r2_per_metric"]),
            "all_val_r2_per_metric": dumps_json(summary["all_val_r2_per_metric"]),

            "all_train_mean_r2": dumps_json(summary["all_train_mean_r2"]),
            "all_val_mean_r2": dumps_json(summary["all_val_mean_r2"]),

            "all_train_loss": dumps_json(summary["all_train_loss"]),
            "all_train_mse_loss": dumps_json(summary["all_train_mse_loss"]),
            "all_train_mae_loss": dumps_json(summary["all_train_mae_loss"]),

            "all_val_loss": dumps_json(summary["all_val_loss"]),
            "all_val_mse_loss": dumps_json(summary["all_val_mse_loss"]),
            "all_val_mae_loss": dumps_json(summary["all_val_mae_loss"]),

            "mean_train_loss": dumps_json(summary["mean_train_loss"]),
            "mean_train_mse_loss": dumps_json(summary["mean_train_mse_loss"]),
            "mean_train_mae_loss": dumps_json(summary["mean_train_mae_loss"]),

            "mean_val_loss": dumps_json(summary["mean_val_loss"]),
            "mean_val_mse_loss": dumps_json(summary["mean_val_mse_loss"]),
            "mean_val_mae_loss": dumps_json(summary["mean_val_mae_loss"]),
            
            "num_folds": summary["num_folds"],

            "std_train_r2": summary["std_train_r2"],
            "std_val_r2": summary["std_val_r2"],
            "var_train_r2": summary["var_train_r2"],
            "var_val_r2": summary["var_val_r2"],

            "std_train_r2_per_metric": dumps_json(summary["std_train_r2_per_metric"]),
            "std_val_r2_per_metric": dumps_json(summary["std_val_r2_per_metric"]),
            "var_train_r2_per_metric": dumps_json(summary["var_train_r2_per_metric"]),
            "var_val_r2_per_metric": dumps_json(summary["var_val_r2_per_metric"]),

            "std_train_loss": dumps_json(summary["std_train_loss"]),
            "std_train_mse_loss": dumps_json(summary["std_train_mse_loss"]),
            "std_train_mae_loss": dumps_json(summary["std_train_mae_loss"]),

            "std_val_loss": dumps_json(summary["std_val_loss"]),
            "std_val_mse_loss": dumps_json(summary["std_val_mse_loss"]),
            "std_val_mae_loss": dumps_json(summary["std_val_mae_loss"]),

            "var_train_loss": dumps_json(summary["var_train_loss"]),
            "var_train_mse_loss": dumps_json(summary["var_train_mse_loss"]),
            "var_train_mae_loss": dumps_json(summary["var_train_mae_loss"]),

            "var_val_loss": dumps_json(summary["var_val_loss"]),
            "var_val_mse_loss": dumps_json(summary["var_val_mse_loss"]),
            "var_val_mae_loss": dumps_json(summary["var_val_mae_loss"]),
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(save_path, index=False)


def train_final_model_on_full_data(
    cfg,
    best_hyper: Dict[str, Any],
    full_data_dict: Dict[str, Any],
    input_shape: tuple[int, int],
    device: torch.device,
    logger,
):
    batch_size = int(best_hyper.get("batch_size", cfg.dataset.batch_size))

    full_loader_dict = build_full_train_dataloader(
        x_device=full_data_dict["x_device"],
        y_logged=full_data_dict["y_logged"],
        batch_size=batch_size,
        num_workers=cfg.dataset.num_workers,
        normalize_x=cfg.dataset.normalize_x,
        normalize_y=cfg.dataset.normalize_y,
        drop_last=False,
    )

    model = build_model(
        model_name=best_hyper["name"],
        model_hyper_parameters=best_hyper,
        input_shape=input_shape,
        device_messages=full_data_dict["device_messages"],
        device_level_attn_mask=full_data_dict.get("device_level_attn_mask", None),
    ).to(device)

    optimizer = build_optimizer(cfg, model, best_hyper)
    scheduler = build_scheduler(cfg, optimizer)

    train_loader = full_loader_dict["train_loader"]

    epochs = int(cfg.exp.epochs)
    alpha = float(cfg.exp.alpha)
    log_interval = int(cfg.exp.log_interval)

    progress = tqdm(
        range(epochs),
        desc="Final full-source training",
        leave=True,
    )

    train_loss_curve = []
    train_mse_curve = []
    train_mae_curve = []

    for epoch in progress:
        train_metrics = run_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            alpha=alpha,
            train=True,
        )

        if scheduler is not None:
            scheduler.step()

        train_loss_curve.append(train_metrics["loss"])
        train_mse_curve.append(train_metrics["mse_loss"])
        train_mae_curve.append(train_metrics["mae_loss"])

        progress.set_postfix(
            {
                "loss": f"{train_metrics['loss']:.4f}",
            }
        )

        if (epoch + 1) % log_interval == 0 or (epoch + 1) == epochs:
            logger.info(
                f"Final Train | Epoch {epoch + 1}/{epochs} | "
                f"loss={train_metrics['loss']:.6f}"
            )

    return model, full_loader_dict, {
        "train_loss": train_loss_curve,
        "train_mse_loss": train_mse_curve,
        "train_mae_loss": train_mae_curve,
    }
    
def _stat_ddof(num_items: int) -> int:
    return 1 if num_items > 1 else 0


def std_curve(curves: List[List[float]]) -> List[float]:
    arr = np.asarray(curves, dtype=np.float64)
    return arr.std(axis=0, ddof=_stat_ddof(arr.shape[0])).tolist()


def var_curve(curves: List[List[float]]) -> List[float]:
    arr = np.asarray(curves, dtype=np.float64)
    return arr.var(axis=0, ddof=_stat_ddof(arr.shape[0])).tolist()