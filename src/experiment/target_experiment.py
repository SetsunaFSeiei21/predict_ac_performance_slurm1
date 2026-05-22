import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.base import build_model, get_logger
from src.base.data_engine import (
    build_dataloaders_from_arrays,
    build_full_dataset_arrays,
)
from src.utils.experiment_utils import (
    build_optimizer,
    build_scheduler,
    evaluate_r2,
    get_device,
    run_one_epoch,
    set_seed,
)


def make_train_valid_indices(
    num_samples: int,
    train_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    assert 0.0 < train_ratio < 1.0, "train_ratio must be in (0, 1)"

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)

    train_num = int(num_samples * train_ratio)

    train_indices = indices[:train_num]
    valid_indices = indices[train_num:]

    if len(train_indices) == 0:
        raise ValueError("Train split is empty.")

    if len(valid_indices) == 0:
        raise ValueError("Valid split is empty.")

    return train_indices, valid_indices


def resolve_source_checkpoint_path(
    cfg,
    model_name: str,
    circuit_name: str,
) -> Path:
    source_checkpoint_path = cfg.get("source_checkpoint_path", None)

    if source_checkpoint_path is not None:
        return Path(source_checkpoint_path)

    source_weight_root = Path(cfg.source_weight_root)

    return (
        source_weight_root
        / model_name
        / circuit_name
        / "source"
        / f"{model_name}_{circuit_name}_source.pt"
    )


def load_source_state_dict(
    model: torch.nn.Module,
    source_state_dict: Dict[str, torch.Tensor],
    strict: bool,
    logger,
) -> None:
    if strict:
        model.load_state_dict(source_state_dict, strict=True)
        logger.info("Loaded source state_dict with strict=True.")
        return

    target_state_dict = model.state_dict()
    compatible_state_dict = {}
    skipped = []

    for key, value in source_state_dict.items():
        if key not in target_state_dict:
            skipped.append((key, "missing_in_target_model"))
            continue

        if target_state_dict[key].shape != value.shape:
            skipped.append(
                (
                    key,
                    f"shape_mismatch source={tuple(value.shape)} "
                    f"target={tuple(target_state_dict[key].shape)}",
                )
            )
            continue

        compatible_state_dict[key] = value

    target_state_dict.update(compatible_state_dict)
    model.load_state_dict(target_state_dict, strict=True)

    logger.info(
        f"Loaded compatible source parameters: "
        f"{len(compatible_state_dict)} / {len(source_state_dict)}"
    )

    if len(skipped) > 0:
        logger.info(f"Skipped parameters: {skipped}")


def call_model_initial_weight(
    model: torch.nn.Module,
    target_init_cfg,
    logger,
) -> Dict[str, Any]:
    if not hasattr(model, "_initial_weight"):
        raise AttributeError(
            f"{model.__class__.__name__} does not implement _initial_weight()."
        )

    target_init_dict = OmegaConf.to_container(
        target_init_cfg,
        resolve=True,
    )

    if target_init_dict is None:
        target_init_dict = {}

    logger.info(
        "Calling model._initial_weight with config:\n"
        + json.dumps(target_init_dict, indent=4, ensure_ascii=False)
    )

    model._initial_weight(**target_init_dict)

    return target_init_dict


def save_target_result_to_excel(
    result_dict: Dict[str, Any],
    save_path: str,
) -> None:
    row = {
        "source_checkpoint_path": result_dict["source_checkpoint_path"],
        "source_best_hyper_parameters": json.dumps(
            result_dict["source_best_hyper_parameters"],
            ensure_ascii=False,
        ),
        "target_train_hyper": json.dumps(
            result_dict["target_train_hyper"],
            ensure_ascii=False,
        ),
        "target_init": json.dumps(
            result_dict["target_init"],
            ensure_ascii=False,
        ),
        "best_epoch": result_dict["best_epoch"],
        "best_val_mean_r2": result_dict["best_val_mean_r2"],
        "best_val_r2_per_metric": json.dumps(
            result_dict["best_val_r2_per_metric"],
            ensure_ascii=False,
        ),
        "train_loss_curve": json.dumps(
            result_dict["train_loss_curve"],
            ensure_ascii=False,
        ),
        "train_mse_curve": json.dumps(
            result_dict["train_mse_curve"],
            ensure_ascii=False,
        ),
        "train_mae_curve": json.dumps(
            result_dict["train_mae_curve"],
            ensure_ascii=False,
        ),
        "val_loss_curve": json.dumps(
            result_dict["val_loss_curve"],
            ensure_ascii=False,
        ),
        "val_mse_curve": json.dumps(
            result_dict["val_mse_curve"],
            ensure_ascii=False,
        ),
        "val_mae_curve": json.dumps(
            result_dict["val_mae_curve"],
            ensure_ascii=False,
        ),
        "val_mean_r2_curve": json.dumps(
            result_dict["val_mean_r2_curve"],
            ensure_ascii=False,
        ),
        "log_y_columns": json.dumps(
            result_dict["log_y_columns"],
            ensure_ascii=False,
        ),
        "outlier_info": json.dumps(
            result_dict["outlier_info"],
            ensure_ascii=False,
        ),
    }

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([row]).to_excel(save_path, index=False)


@hydra.main(
    version_base="1.3",
    config_path=os.path.join(os.getcwd(), "configs"),
    config_name="target",
)
def main(cfg) -> None:
    set_seed(int(cfg.seed))

    model_name = cfg.model.name
    circuit_name = cfg.dataset.circuit_name

    result_root = Path(cfg.result_root)
    weight_root = Path(cfg.weight_root)
    log_root = Path(cfg.log_root)

    result_dir = result_root / model_name / circuit_name / "target"
    weight_dir = weight_root / model_name / circuit_name / "target"
    log_dir = log_root / model_name / circuit_name / "target"

    result_dir.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(
        log_file_path=str(log_dir / "target_experiment.log"),
        logger_name=f"{model_name}_{circuit_name}_target",
    )

    device = get_device()

    logger.info(f"Using device: {device}")
    logger.info("Loaded config:")
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    source_checkpoint_path = resolve_source_checkpoint_path(
        cfg=cfg,
        model_name=model_name,
        circuit_name=circuit_name,
    )

    if not source_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Source checkpoint not found: {source_checkpoint_path}"
        )

    logger.info(f"Loading source checkpoint: {source_checkpoint_path}")

    checkpoint = torch.load(
        source_checkpoint_path,
        map_location=device,
    )

    if "best_hyper_parameters" not in checkpoint:
        raise KeyError("Source checkpoint missing key: best_hyper_parameters")

    if "model_state_dict" not in checkpoint:
        raise KeyError("Source checkpoint missing key: model_state_dict")

    source_best_hyper = checkpoint["best_hyper_parameters"]
    source_state_dict = checkpoint["model_state_dict"]

    source_model_name = source_best_hyper.get("name", None)

    if source_model_name is None:
        raise KeyError("source best_hyper_parameters missing key: name")

    if source_model_name != model_name:
        raise ValueError(
            f"Model name mismatch: cfg.model.name={model_name}, "
            f"source checkpoint model name={source_model_name}"
        )

    logger.info(
        "Loaded source best hyperparameters:\n"
        + json.dumps(source_best_hyper, indent=4, ensure_ascii=False)
    )

    full_data_dict = build_full_dataset_arrays(
        cfg=cfg,
        logger=logger,
    )

    x_device = full_data_dict["x_device"]
    y_logged = full_data_dict["y_logged"]

    input_shape = (
        int(x_device.shape[1]),
        int(x_device.shape[2]),
    )

    logger.info(f"Target input shape: {input_shape}")
    logger.info(f"Target y shape: {y_logged.shape}")

    train_indices, valid_indices = make_train_valid_indices(
        num_samples=len(x_device),
        train_ratio=float(cfg.dataset.train_ratio),
        seed=int(cfg.seed),
    )

    batch_size = int(source_best_hyper.get("batch_size", cfg.dataset.batch_size))

    loader_dict = build_dataloaders_from_arrays(
        x_device=x_device,
        y_logged=y_logged,
        train_indices=train_indices,
        valid_indices=valid_indices,
        batch_size=batch_size,
        num_workers=cfg.dataset.num_workers,
        normalize_x=cfg.dataset.normalize_x,
        normalize_y=cfg.dataset.normalize_y,
        drop_last=False,
    )

    model = build_model(
        model_name=source_best_hyper["name"],
        model_hyper_parameters=source_best_hyper,
        input_shape=input_shape,
        device_messages=full_data_dict["device_messages"],
        device_level_attn_mask=full_data_dict.get("device_level_attn_mask", None),
    ).to(device)

    load_source_state_dict(
        model=model,
        source_state_dict=source_state_dict,
        strict=bool(cfg.strict_load_state_dict),
        logger=logger,
    )

    target_init_dict = call_model_initial_weight(
        model=model,
        target_init_cfg=cfg.target_init,
        logger=logger,
    )

    target_train_hyper = deepcopy(source_best_hyper)
    target_train_hyper["lr"] = float(cfg.exp.lr)
    target_train_hyper["weight_decay"] = float(cfg.exp.weight_decay)

    optimizer = build_optimizer(
        cfg=cfg,
        model=model,
        hyper=target_train_hyper,
    )

    scheduler = build_scheduler(
        cfg=cfg,
        optimizer=optimizer,
    )

    train_loader = loader_dict["train_loader"]
    valid_loader = loader_dict["valid_loader"]

    epochs = int(cfg.exp.epochs)
    alpha = float(cfg.exp.alpha)
    log_interval = int(cfg.exp.log_interval)

    train_loss_curve = []
    train_mse_curve = []
    train_mae_curve = []

    val_loss_curve = []
    val_mse_curve = []
    val_mae_curve = []
    val_mean_r2_curve = []

    best_epoch = -1
    best_val_mean_r2 = -1e18
    best_val_r2_per_metric = None
    best_state_dict = None

    progress = tqdm(
        range(epochs),
        desc="Target fine-tuning",
        leave=True,
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

        val_r2 = evaluate_r2(
            model=model,
            data_loader=valid_loader,
            y_scaler=loader_dict["y_scaler"],
            log_y_indices=full_data_dict["log_y_indices"],
            device=device,
        )

        train_loss_curve.append(train_metrics["loss"])
        train_mse_curve.append(train_metrics["mse_loss"])
        train_mae_curve.append(train_metrics["mae_loss"])

        val_loss_curve.append(val_metrics["loss"])
        val_mse_curve.append(val_metrics["mse_loss"])
        val_mae_curve.append(val_metrics["mae_loss"])
        val_mean_r2_curve.append(val_r2["mean_r2"])

        if val_r2["mean_r2"] > best_val_mean_r2:
            best_epoch = epoch + 1
            best_val_mean_r2 = val_r2["mean_r2"]
            best_val_r2_per_metric = val_r2["r2_per_metric"]
            best_state_dict = {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            }

        progress.set_postfix(
            {
                "train_loss": f"{train_metrics['loss']:.4f}",
                "val_loss": f"{val_metrics['loss']:.4f}",
                "val_r2": f"{val_r2['mean_r2']:.4f}",
            }
        )

        if (epoch + 1) % log_interval == 0 or (epoch + 1) == epochs:
            logger.info(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train_loss={train_metrics['loss']:.6f} | "
                f"val_loss={val_metrics['loss']:.6f} | "
                f"val_mean_r2={val_r2['mean_r2']:.6f}"
            )

    if best_state_dict is None:
        raise RuntimeError("No target checkpoint was selected.")

    result_dict = {
        "source_checkpoint_path": str(source_checkpoint_path),
        "source_best_hyper_parameters": source_best_hyper,

        "target_train_hyper": target_train_hyper,
        "target_init": target_init_dict,

        "best_epoch": best_epoch,
        "best_val_mean_r2": best_val_mean_r2,
        "best_val_r2_per_metric": best_val_r2_per_metric,

        "train_loss_curve": train_loss_curve,
        "train_mse_curve": train_mse_curve,
        "train_mae_curve": train_mae_curve,

        "val_loss_curve": val_loss_curve,
        "val_mse_curve": val_mse_curve,
        "val_mae_curve": val_mae_curve,
        "val_mean_r2_curve": val_mean_r2_curve,

        "log_y_indices": full_data_dict["log_y_indices"],
        "log_y_columns": full_data_dict["log_y_columns"],
        "y_columns": full_data_dict["y_columns"],
        "device_order": full_data_dict["device_order"],
        "device_feature_columns": full_data_dict["device_feature_columns"],
        "outlier_info": full_data_dict["outlier_info"],
    }

    with open(result_dir / "target_result.json", "w", encoding="utf-8") as f:
        json.dump(
            result_dict,
            f,
            indent=4,
            ensure_ascii=False,
        )

    save_target_result_to_excel(
        result_dict=result_dict,
        save_path=str(result_dir / "target_result.xlsx"),
    )

    target_weight_path = weight_dir / f"{model_name}_{circuit_name}_target.pt"

    torch.save(
        {
            "model_state_dict": best_state_dict,

            "source_checkpoint_path": str(source_checkpoint_path),
            "source_best_hyper_parameters": source_best_hyper,

            "target_train_hyper": target_train_hyper,
            "target_result": result_dict,

            "x_scaler": loader_dict["x_scaler"],
            "y_scaler": loader_dict["y_scaler"],

            "log_y_indices": full_data_dict["log_y_indices"],
            "log_y_columns": full_data_dict["log_y_columns"],
            "y_columns": full_data_dict["y_columns"],

            "device_messages": full_data_dict["device_messages"],
            "device_order": full_data_dict["device_order"],
            "device_feature_columns": full_data_dict["device_feature_columns"],

            "target_init": target_init_dict,
            "outlier_info": full_data_dict["outlier_info"],
        },
        target_weight_path,
    )

    logger.info(f"Best target epoch: {best_epoch}")
    logger.info(f"Best target val mean R2: {best_val_mean_r2:.6f}")
    logger.info(f"Target checkpoint saved to: {target_weight_path}")


if __name__ == "__main__":
    main()