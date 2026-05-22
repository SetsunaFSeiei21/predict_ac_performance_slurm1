import numpy as np
from typing import Tuple

class IQRFilter:
    
    def __init__(self, iqr_factor: float = 3.0, use_x: bool = False, use_y: bool = True, eps: float = 1e-12):
        
        assert use_x or use_y, "至少一个应该被指定为True"
        self.iqr_factor = iqr_factor
        self.use_x = use_x
        self.use_y = use_y
        self.eps = eps
        self.lower_bound = None
        self.upper_bound = None
        
    def _concat_selected(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        
        arrays = []
        if self.use_x:
            arrays.append(x)
        if self.use_y:
            arrays.append(y)
        
        return np.concatenate(arrays, axis=1)
    
    def fit(self, x: np.ndarray, y: np.ndarray) -> "IQRFilter":
        
        data = self._concat_selected(x, y)
        q1 = np.quantile(data, 0.25, axis=0, keepdims=True)
        q3 = np.quantile(data, 0.75, axis=0, keepdims=True)
        iqr = q3 - q1
        lower_bound = q1 - self.iqr_factor * iqr
        upper_bound = q3 + self.iqr_factor * iqr
        small_iqr_mask = iqr < self.eps
        lower_bound = np.where(small_iqr_mask, -np.inf, lower_bound)
        upper_bound = np.where(small_iqr_mask, np.inf, upper_bound)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
        return self
    
    def get_valid_mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        
        if self.lower_bound is None or self.upper_bound is None:
            raise RuntimeError("IQRFilter has not been filtered.")
        data = self._concat_selected(x, y)
        valid_mask = np.all((data >= self.lower_bound) & (data <= self.upper_bound), axis=1)
        
        return valid_mask
    
    def transform(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        valid_mask = self.get_valid_mask(x, y)
        return x[valid_mask], y[valid_mask], valid_mask