import json, os
import numpy as np
import pandas as pd

SOURCE_DF_FILE_NAME = "pretrain_design_features.csv"
SOURCE_TARGET_FILE_NAME = "pretrain_targets.csv"
TARGET_DF_FILE_NAME = "target_design_features.csv"
TARGET_TARGET_FILE_NAME = "target_targets.csv"
PERFORMANCE_ORDER = ["SR_Rise_V_us", "LoopGain_dB", "LoopUBW_MHz", "LoopPM_deg", "CMRR_DC_dB"]

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
    if os.path.exists(design_features_file_path) and os.path.exists(targets_file_path):
        print("文件已被处理。")
        return
    else:
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
        nan_idx = np.isnan(np_performance).any(axis=1)
        valid_idx = ~nan_idx
        np_design_variables = np_design_variables[valid_idx]
        np_performance = np_performance[valid_idx]
        design_variables_order = list(raw_data[0]['design_variables'].keys())
        df_design_variables = pd.DataFrame(np_design_variables, columns=design_variables_order)
        df_performance = pd.DataFrame(np_performance, columns=PERFORMANCE_ORDER)
        df_design_variables.to_csv(design_features_file_path, index=False)
        df_performance.to_csv(targets_file_path, index=False)

        print(f"原始样本数: {len(raw_data)}")
        print(f"有效样本数: {valid_idx.sum()}")
        print(f"无效样本数: {nan_idx.sum()}")
        print(f"Saved: {design_features_file_path}")
        print(f"Saved: {targets_file_path}")
