import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.dataset import build_ac_dataloaders
from src.dataset.ac_dataset import ACDataset
from src.dataset.dataloader import (
    SOURCE_DF_FILE_NAME,
    SOURCE_TARGET_FILE_NAME,
    TARGET_DF_FILE_NAME,
    TARGET_TARGET_FILE_NAME,
)
from src.dataset.device_features import (
    DEVICE_FEATURE_COLUMNS,
    build_device_features_from_dataframe,
    get_device_order,
    load_device_messages,
)
from src.dataset.outlier import IQRFilter
from src.dataset.scaler import StandardScaler
from src.dataset.transforms import (
    apply_log_by_indices,
    fit_auto_log_y_columns,
)


def build_dataset(cfg) -> Dict[str, Any]:

    dataset_cfg = cfg.dataset
    circuit_dir = os.path.join(cfg.meta_path, dataset_cfg.circuit_name)

    data_dict = build_ac_dataloaders(
        circuit_dir=circuit_dir,
        mission_type=dataset_cfg.mission_type,
        batch_size=dataset_cfg.batch_size,
        num_workers=dataset_cfg.num_workers,
        train_ratio=dataset_cfg.train_ratio,
        seed=cfg.seed,

        normalize_x=dataset_cfg.normalize_x,
        normalize_y=dataset_cfg.normalize_y,

        auto_log_y=dataset_cfg.auto_log_y,
        log_y_ratio_threshold=dataset_cfg.log_y_ratio_threshold,

        remove_outlier=dataset_cfg.remove_outlier,
        iqr_factor=dataset_cfg.iqr_factor,
        outlier_use_x=dataset_cfg.outlier_use_x,
        outlier_use_y=dataset_cfg.outlier_use_y,
    )

    return data_dict


def get_circuit_dir(cfg) -> str:
    return os.path.join(cfg.meta_path, cfg.dataset.circuit_name)


def get_csv_paths(circuit_dir: str, mission_type: str) -> Tuple[str, str]:
    mission_dir = os.path.join(circuit_dir, mission_type)

    if mission_type == "source":
        design_features_csv = os.path.join(mission_dir, SOURCE_DF_FILE_NAME)
        targets_csv = os.path.join(mission_dir, SOURCE_TARGET_FILE_NAME)
    elif mission_type == "target":
        design_features_csv = os.path.join(mission_dir, TARGET_DF_FILE_NAME)
        targets_csv = os.path.join(mission_dir, TARGET_TARGET_FILE_NAME)
    else:
        raise ValueError(f"Unsupported mission_type: {mission_type}")

    if not os.path.exists(design_features_csv):
        raise FileNotFoundError(f"Missing design feature file: {design_features_csv}")

    if not os.path.exists(targets_csv):
        raise FileNotFoundError(f"Missing target file: {targets_csv}")

    return design_features_csv, targets_csv

def load_device_level_attn_mask(circuit_dir: str) -> np.ndarray:
    mask_path = os.path.join(circuit_dir, "device_level_netlist.npy")

    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Missing device-level attention mask: {mask_path}")

    attn_mask = np.load(mask_path)

    if attn_mask.ndim != 2:
        raise ValueError(f"device_level_attn_mask must be 2D, got {attn_mask.shape}")

    if attn_mask.shape[0] != attn_mask.shape[1]:
        raise ValueError(f"device_level_attn_mask must be square, got {attn_mask.shape}")

    return attn_mask.astype(np.float32)

def build_full_dataset_arrays(cfg, logger=None) -> Dict[str, Any]:
    """
    构造清洗后的全量数据数组。

    顺序：
        raw design_df / target_df
        -> Y auto log
        -> IQR filter
        -> raw design_df_filtered 构造 x_device
        -> 返回 x_device / y_logged

    注意：
        这里不做 train/valid split，不做 z-score。
        z-score 应该在每个 fold 内单独 fit。
    """
    dataset_cfg = cfg.dataset
    circuit_dir = get_circuit_dir(cfg)

    design_features_csv, targets_csv = get_csv_paths(
        circuit_dir=circuit_dir,
        mission_type=dataset_cfg.mission_type,
    )

    design_df = pd.read_csv(design_features_csv)
    target_df = pd.read_csv(targets_csv)

    design_columns = list(design_df.columns)
    y_columns = list(target_df.columns)

    y_raw = target_df.values.astype(np.float64)
    y_logged = y_raw.copy()

    log_y_indices: List[int] = []
    log_y_columns: List[str] = []

    if dataset_cfg.auto_log_y:
        log_y_indices, log_y_columns = fit_auto_log_y_columns(
            y_train=y_logged,
            y_columns=y_columns,
            ratio_threshold=dataset_cfg.log_y_ratio_threshold,
        )

        y_logged = apply_log_by_indices(
            y=y_logged,
            log_indices=log_y_indices,
        )

    iqr_filter: Optional[IQRFilter] = None
    outlier_info: Dict[str, int] = {
        "all_removed": 0,
    }

    if dataset_cfg.remove_outlier:
        if dataset_cfg.outlier_use_x:
            raise ValueError(
                "Current pipeline assumes outlier_use_x=False. "
                "Use target-only IQR filtering first."
            )

        iqr_filter = IQRFilter(
            iqr_factor=dataset_cfg.iqr_factor,
            use_x=dataset_cfg.outlier_use_x,
            use_y=dataset_cfg.outlier_use_y,
        )

        x_placeholder = design_df.values.astype(np.float64)
        iqr_filter.fit(x_placeholder, y_logged)

        before_num = len(design_df)

        _, y_logged_filtered, valid_mask = iqr_filter.transform(
            x=x_placeholder,
            y=y_logged,
        )

        design_df_filtered = design_df.loc[valid_mask].reset_index(drop=True)
        outlier_info["all_removed"] = before_num - len(design_df_filtered)

    else:
        design_df_filtered = design_df.reset_index(drop=True)
        y_logged_filtered = y_logged

    if len(design_df_filtered) != len(y_logged_filtered):
        raise ValueError(
            f"Filtered X/Y sample num mismatch: "
            f"{len(design_df_filtered)} vs {len(y_logged_filtered)}"
        )

    device_messages = load_device_messages(circuit_dir)
    device_order = get_device_order(device_messages)
    
    device_level_attn_mask = load_device_level_attn_mask(circuit_dir)

    x_device = build_device_features_from_dataframe(
        design_df=design_df_filtered,
        device_messages=device_messages,
    )

    if logger is not None:
        logger.info(f"Raw design_df shape: {design_df.shape}")
        logger.info(f"Raw target_df shape: {target_df.shape}")
        logger.info(f"Cleaned x_device shape: {x_device.shape}")
        logger.info(f"Cleaned y_logged shape: {y_logged_filtered.shape}")
        logger.info(f"Log Y columns: {log_y_columns}")
        logger.info(f"Outlier info: {outlier_info}")

    return {
        "x_device": x_device.astype(np.float64),
        "y_logged": y_logged_filtered.astype(np.float64),

        "design_columns": design_columns,
        "y_columns": y_columns,

        "device_messages": device_messages,
        "device_order": device_order,
        "device_feature_columns": DEVICE_FEATURE_COLUMNS,
        "device_level_attn_mask": device_level_attn_mask,

        "log_y_indices": log_y_indices,
        "log_y_columns": log_y_columns,

        "iqr_filter": iqr_filter,
        "outlier_info": outlier_info,
    }


def make_kfold_indices(
    num_samples: int,
    k_fold: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    assert k_fold > 1, "k_fold must be larger than 1"
    assert num_samples >= k_fold, "num_samples must be no less than k_fold"

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)

    fold_sizes = np.full(k_fold, num_samples // k_fold, dtype=int)
    fold_sizes[: num_samples % k_fold] += 1

    folds = []
    current = 0

    for fold_size in fold_sizes:
        valid_idx = indices[current: current + fold_size]
        train_idx = np.concatenate(
            [
                indices[:current],
                indices[current + fold_size:],
            ],
            axis=0,
        )

        folds.append((train_idx, valid_idx))
        current += fold_size

    return folds


def fit_transform_device_features_with_2d_scaler(
    x_train: np.ndarray,
    x_valid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], StandardScaler]:
    if x_train.ndim != 3:
        raise ValueError(f"Expected x_train shape [N, D, F], got {x_train.shape}")

    train_shape = x_train.shape
    x_train_2d = x_train.reshape(train_shape[0], -1)

    x_scaler = StandardScaler().fit(x_train_2d)
    x_train_2d = x_scaler.transform(x_train_2d)
    x_train = x_train_2d.reshape(train_shape)

    if x_valid is None:
        return x_train, None, x_scaler

    if x_valid.ndim != 3:
        raise ValueError(f"Expected x_valid shape [N, D, F], got {x_valid.shape}")

    valid_shape = x_valid.shape
    x_valid_2d = x_valid.reshape(valid_shape[0], -1)
    x_valid_2d = x_scaler.transform(x_valid_2d)
    x_valid = x_valid_2d.reshape(valid_shape)

    return x_train, x_valid, x_scaler


def build_dataloaders_from_arrays(
    x_device: np.ndarray,
    y_logged: np.ndarray,
    train_indices: np.ndarray,
    valid_indices: Optional[np.ndarray],
    batch_size: int,
    num_workers: int,
    normalize_x: bool = True,
    normalize_y: bool = True,
    drop_last: bool = False,
) -> Dict[str, Any]:
    x_train = x_device[train_indices]
    y_train = y_logged[train_indices]

    x_valid = None
    y_valid = None

    if valid_indices is not None:
        x_valid = x_device[valid_indices]
        y_valid = y_logged[valid_indices]

    x_scaler = None
    y_scaler = None

    if normalize_x:
        x_train, x_valid, x_scaler = fit_transform_device_features_with_2d_scaler(
            x_train=x_train,
            x_valid=x_valid,
        )

    if normalize_y:
        y_scaler = StandardScaler().fit(y_train)
        y_train = y_scaler.transform(y_train)

        if y_valid is not None:
            y_valid = y_scaler.transform(y_valid)

    train_dataset = ACDataset(x_train, y_train)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    result = {
        "train_loader": train_loader,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "train_size": len(train_dataset),
    }

    if valid_indices is not None:
        valid_dataset = ACDataset(x_valid, y_valid)

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

        result.update(
            {
                "valid_loader": valid_loader,
                "valid_size": len(valid_dataset),
            }
        )

    return result


def build_fold_dataloaders(
    x_device: np.ndarray,
    y_logged: np.ndarray,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    normalize_x: bool = True,
    normalize_y: bool = True,
    drop_last: bool = False,
) -> Dict[str, Any]:
    return build_dataloaders_from_arrays(
        x_device=x_device,
        y_logged=y_logged,
        train_indices=train_indices,
        valid_indices=valid_indices,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize_x=normalize_x,
        normalize_y=normalize_y,
        drop_last=drop_last,
    )


def build_full_train_dataloader(
    x_device: np.ndarray,
    y_logged: np.ndarray,
    batch_size: int,
    num_workers: int,
    normalize_x: bool = True,
    normalize_y: bool = True,
    drop_last: bool = False,
) -> Dict[str, Any]:
    all_indices = np.arange(len(x_device))

    return build_dataloaders_from_arrays(
        x_device=x_device,
        y_logged=y_logged,
        train_indices=all_indices,
        valid_indices=None,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize_x=normalize_x,
        normalize_y=normalize_y,
        drop_last=drop_last,
    )