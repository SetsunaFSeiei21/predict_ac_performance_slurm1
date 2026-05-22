import os
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .ac_dataset import ACDataset
from .scaler import StandardScaler
from .transforms import (
    fit_auto_log_y_columns,
    apply_log_by_indices,
    inverse_log_by_indices,
)
from .outlier import IQRFilter
from .device_features import (
    DEVICE_FEATURE_COLUMNS,
    build_device_features_from_dataframe,
    get_device_order,
    load_device_messages,
)


SOURCE_DF_FILE_NAME = "pretrain_design_features.csv"
SOURCE_TARGET_FILE_NAME = "pretrain_targets.csv"
TARGET_DF_FILE_NAME = "target_design_features.csv"
TARGET_TARGET_FILE_NAME = "target_targets.csv"


def split_train_valid_numpy_data(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    assert len(x) == len(y), "x and y must have the same number of samples."
    assert 0.0 < train_ratio < 1.0

    total_num = len(x)
    train_num = int(total_num * train_ratio)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_num, generator=generator).numpy()

    train_idx = indices[:train_num]
    valid_idx = indices[train_num:]

    return (
        x[train_idx],
        y[train_idx],
        x[valid_idx],
        y[valid_idx],
    )


def _check_non_empty_split(
    x_train: np.ndarray,
    x_valid: np.ndarray,
) -> None:
    if len(x_train) == 0:
        raise ValueError("Train set is empty after preprocessing.")

    if len(x_valid) == 0:
        raise ValueError("Valid set is empty after preprocessing.")


def _fit_transform_device_features_with_2d_scaler(
    x_train: np.ndarray,
    x_valid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:

    if x_train.ndim != 3 or x_valid.ndim != 3:
        raise ValueError(
            f"Expected 3D device features, got x_train={x_train.shape}, x_valid={x_valid.shape}"
        )

    train_shape = x_train.shape
    valid_shape = x_valid.shape

    x_train_2d = x_train.reshape(train_shape[0], -1)
    x_valid_2d = x_valid.reshape(valid_shape[0], -1)

    x_scaler = StandardScaler().fit(x_train_2d)

    x_train_2d = x_scaler.transform(x_train_2d)
    x_valid_2d = x_scaler.transform(x_valid_2d)

    x_train = x_train_2d.reshape(train_shape)
    x_valid = x_valid_2d.reshape(valid_shape)

    return x_train, x_valid, x_scaler


def build_ac_dataloaders(
    circuit_dir: str,
    mission_type: str,
    batch_size: int,
    num_workers: int = 8,
    train_ratio: float = 0.9,
    seed: int = 42,
    normalize_x: bool = True,
    normalize_y: bool = True,
    auto_log_y: bool = True,
    log_y_ratio_threshold: float = 1e4,
    remove_outlier: bool = True,
    iqr_factor: float = 3.0,
    outlier_use_x: bool = False,
    outlier_use_y: bool = True,
    drop_last: bool = False,
) -> Dict[str, Any]:

    assert mission_type in ["source", "target"], (f"mission_type must be 'source' or 'target', but got {mission_type}")

    if outlier_use_x:
        raise ValueError(
            "Current version assumes outlier_use_x=False. "
            "Use outlier_use_y=True to remove outliers based on targets only."
        )

    mission_dir = os.path.join(circuit_dir, mission_type)

    if mission_type == "source":
        design_features_csv = os.path.join(mission_dir, SOURCE_DF_FILE_NAME)
        targets_csv = os.path.join(mission_dir, SOURCE_TARGET_FILE_NAME)
    else:
        design_features_csv = os.path.join(mission_dir, TARGET_DF_FILE_NAME)
        targets_csv = os.path.join(mission_dir, TARGET_TARGET_FILE_NAME)

    if not os.path.exists(design_features_csv):
        raise FileNotFoundError(f"Missing design feature file: {design_features_csv}")

    if not os.path.exists(targets_csv):
        raise FileNotFoundError(f"Missing target file: {targets_csv}")

    # 1. 读取二维原始数据
    design_df = pd.read_csv(design_features_csv)
    target_df = pd.read_csv(targets_csv)

    design_columns = list(design_df.columns)
    y_columns = list(target_df.columns)

    y_raw = target_df.values.astype(np.float64)

    print(f"[Raw Data] design_df shape: {design_df.shape}")
    print(f"[Raw Data] y_raw shape: {y_raw.shape}")

    # 2. 对整个 Y 先判断 log 列，并取 log
    log_y_indices: List[int] = []
    log_y_columns: List[str] = []

    y_logged = y_raw.copy()

    if auto_log_y:
        log_y_indices, log_y_columns = fit_auto_log_y_columns(
            y_train=y_logged,
            y_columns=y_columns,
            ratio_threshold=log_y_ratio_threshold,
        )

        print(f"[Auto Log Y] columns: {log_y_columns}")

        y_logged = apply_log_by_indices(
            y=y_logged,
            log_indices=log_y_indices,
        )

    iqr_filter: Optional[IQRFilter] = None
    outlier_info: Dict[str, int] = {
        "all_removed": 0,
    }

    if remove_outlier:
        iqr_filter = IQRFilter(
            iqr_factor=iqr_factor,
            use_x=outlier_use_x,
            use_y=outlier_use_y,
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

        print(
            "[IQR Outlier Removal] "
            f"removed: {outlier_info['all_removed']} / {before_num}"
        )

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

    x_device = build_device_features_from_dataframe(
        design_df=design_df_filtered,
        device_messages=device_messages,
    )

    print(f"[Device Features Raw] x_device shape: {x_device.shape}")
    print(f"[Targets Logged] y_logged_filtered shape: {y_logged_filtered.shape}")

    # 5. 划分 train / valid
    x_train, y_train, x_valid, y_valid = split_train_valid_numpy_data(
        x=x_device,
        y=y_logged_filtered,
        train_ratio=train_ratio,
        seed=seed,
    )

    _check_non_empty_split(x_train, x_valid)

    print(f"[Split] train size: {len(x_train)}")
    print(f"[Split] valid size: {len(x_valid)}")

    # 6. X_device 归一化：
    #    flatten -> StandardScaler -> reshape
    x_scaler: Optional[StandardScaler] = None
    y_scaler: Optional[StandardScaler] = None

    if normalize_x:
        x_train, x_valid, x_scaler = _fit_transform_device_features_with_2d_scaler(
            x_train=x_train,
            x_valid=x_valid,
        )

    # 7. Y 归一化
    if normalize_y:
        y_scaler = StandardScaler().fit(y_train)

        y_train = y_scaler.transform(y_train)
        y_valid = y_scaler.transform(y_valid)

    # 8. Dataset / DataLoader
    train_dataset = ACDataset(x_train, y_train)
    valid_dataset = ACDataset(x_valid, y_valid)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,

        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "iqr_filter": iqr_filter,

        "design_columns": design_columns,
        "y_columns": y_columns,

        "device_order": device_order,
        "device_messages": device_messages,
        "device_feature_columns": DEVICE_FEATURE_COLUMNS,

        "log_y_indices": log_y_indices,
        "log_y_columns": log_y_columns,

        "outlier_info": outlier_info,

        "train_size": len(train_dataset),
        "valid_size": len(valid_dataset),
    }


def recover_y_to_original_scale(y: np.ndarray, y_scaler: Optional[StandardScaler], log_y_indices: List[int]) -> np.ndarray:

    y = y.copy()
    if y_scaler is not None:
        y = y_scaler.inverse_transform(y)
    if len(log_y_indices) > 0:
        y = inverse_log_by_indices(y, log_y_indices)

    return y