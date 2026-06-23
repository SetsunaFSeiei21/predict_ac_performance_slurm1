import json, os
import numpy as np
import pandas as pd

SOURCE_DF_FILE_NAME = "pretrain_design_features.csv"
SOURCE_TARGET_FILE_NAME = "pretrain_targets.csv"
TARGET_DF_FILE_NAME = "target_design_features.csv"
TARGET_TARGET_FILE_NAME = "target_targets.csv"
PERFORMANCE_ORDER = ["SR_Rise_V_us", "LoopGain_dB", "LoopUBW_MHz", "LoopPM_deg", "CMRR_DC_dB"]


def clean_xy_csv(design_features_file_path: str, targets_file_path: str) -> None:
    
    design_df = pd.read_csv(design_features_file_path)
    targets_df = pd.read_csv(targets_file_path)

    if len(design_df) != len(targets_df):
        raise ValueError(
            f"X/Y sample number mismatch: "
            f"{design_features_file_path} has {len(design_df)} rows, "
            f"{targets_file_path} has {len(targets_df)} rows."
        )

    design_df = design_df.replace([np.inf, -np.inf], np.nan)
    targets_df = targets_df.replace([np.inf, -np.inf], np.nan)

    valid_mask = ~(
        design_df.isna().any(axis=1) |
        targets_df.isna().any(axis=1)
    )

    before_num = len(design_df)
    after_num = int(valid_mask.sum())
    removed_num = before_num - after_num

    design_df = design_df.loc[valid_mask].reset_index(drop=True)
    targets_df = targets_df.loc[valid_mask].reset_index(drop=True)

    design_df.to_csv(design_features_file_path, index=False)
    targets_df.to_csv(targets_file_path, index=False)

    print(f"[Clean CSV] {design_features_file_path}")
    print(f"[Clean CSV] {targets_file_path}")
    print(f"原始样本数: {before_num}")
    print(f"有效样本数: {after_num}")
    print(f"删除样本数: {removed_num}")


def convert_json2csv(mission_dir: str) -> None:
    
    mission_type = os.path.basename(mission_dir)

    if mission_type == "source":
        design_features_file_path = os.path.join(mission_dir, SOURCE_DF_FILE_NAME)
        targets_file_path = os.path.join(mission_dir, SOURCE_TARGET_FILE_NAME)
        process_file_path = os.path.join(mission_dir, "source_raw_data.json")
    elif mission_type == "target":
        design_features_file_path = os.path.join(mission_dir, TARGET_DF_FILE_NAME)
        targets_file_path = os.path.join(mission_dir, TARGET_TARGET_FILE_NAME)
        process_file_path = os.path.join(mission_dir, "target_raw_data.json")
    else:
        raise ValueError(f"Unsupported mission type: {mission_type}")

    if os.path.exists(design_features_file_path) and os.path.exists(targets_file_path):
        print("CSV files already exist. Cleaning NaN rows...")
        clean_xy_csv(design_features_file_path, targets_file_path)
        return

    if os.path.exists(design_features_file_path) != os.path.exists(targets_file_path):
        raise FileNotFoundError(
            f"Only one CSV file exists. Please make sure both files exist:\n"
            f"  {design_features_file_path}\n"
            f"  {targets_file_path}"
        )

    with open(process_file_path, mode='r', encoding='utf-8') as f:
        raw_data = json.load(f)

    design_variables_lst = []
    performance_lst = []

    for i in range(len(raw_data)):
        tmp_design_variables = raw_data[i]['design_variables']
        design_variables_lst.append(list(tmp_design_variables.values()))

        tmp_performance = raw_data[i]['circuit_performances']
        tmp_performance_lst = []

        for j in PERFORMANCE_ORDER:
            tmp_performance_lst.append(tmp_performance[j][0])

        performance_lst.append(tmp_performance_lst)

    np_design_variables = np.array(design_variables_lst, dtype=np.float64)
    np_performance = np.array(performance_lst, dtype=np.float64)

    design_variables_order = list(raw_data[0]['design_variables'].keys())

    df_design_variables = pd.DataFrame(
        np_design_variables,
        columns=design_variables_order,
    )

    df_performance = pd.DataFrame(
        np_performance,
        columns=PERFORMANCE_ORDER,
    )

    df_design_variables = df_design_variables.replace([np.inf, -np.inf], np.nan)
    df_performance = df_performance.replace([np.inf, -np.inf], np.nan)

    valid_mask = ~(
        df_design_variables.isna().any(axis=1) |
        df_performance.isna().any(axis=1)
    )

    before_num = len(df_design_variables)

    df_design_variables = df_design_variables.loc[valid_mask].reset_index(drop=True)
    df_performance = df_performance.loc[valid_mask].reset_index(drop=True)

    df_design_variables.to_csv(design_features_file_path, index=False)
    df_performance.to_csv(targets_file_path, index=False)

    print(f"原始样本数: {before_num}")
    print(f"有效样本数: {int(valid_mask.sum())}")
    print(f"无效样本数: {int((~valid_mask).sum())}")
    print(f"Saved: {design_features_file_path}")
    print(f"Saved: {targets_file_path}")