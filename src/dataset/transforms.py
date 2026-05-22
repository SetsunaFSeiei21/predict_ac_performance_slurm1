import numpy as np
from typing import List, Tuple, Optional

def fit_auto_log_y_columns(
    y_train: np.ndarray, y_columns: List[str], ratio_threshold: float = 1e3, eps: float = 1e-12, skip_keywords: Optional[Tuple[str, ...]] = None,
) -> Tuple[List[int], List[str]]:
    log_indices = []
    log_columns = []
    
    for col_idx, col_name in enumerate(y_columns):
        lower_col_name = col_name.lower()
        
        if skip_keywords is not None and any(keyword in lower_col_name for keyword in skip_keywords):
            continue
        
        col = y_train[:, col_idx]
        min_value = np.min(col)
        max_value = np.max(col)
        
        if min_value <= 0:
            continue
        
        ratio = max_value / (min_value + eps)
        if ratio > ratio_threshold:
            log_indices.append(col_idx)
            log_columns.append(col_name)
    
    return log_indices, log_columns

def apply_log_by_indices(y: np.ndarray, log_indices: List[int], eps: float = 1e-12) -> np.ndarray:
    
    y = y.copy()
    for idx in log_indices:
        if np.any(y[:, idx] <= 0):
            raise ValueError( f"Column index {idx} contains non-positive values, cannot apply log.")
        y[:, idx] = np.log(y[:,idx] + eps)
    
    return y

def inverse_log_by_indices(y: np.ndarray, log_indices: List[int]) -> np.ndarray:
    
    y = y.copy()
    for idx in log_indices:
        y[:, idx] = np.exp(y[:,idx])
    
    return y