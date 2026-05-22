import numpy as np
from typing import Optional

class StandardScaler:
    
    def __init__(self, eps: float = 1e-9) -> None:
        
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.eps: float = eps
        
    def fit(self, x: np.ndarray) -> "StandardScaler":
        
        self.mean = np.mean(x, axis=0, keepdims=True)
        self.std = np.std(x, axis=0, keepdims=True)
        self.std = np.where(self.std < self.eps, 1.0, self.std)
        return self
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        
        if self.mean is None or self.std is None:
            raise RuntimeError("标准化未进行！")
        return (x - self.mean) / self.std
    
    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        
        if self.mean is None or self.std is None:
            raise RuntimeError("标准化未进行!")
        return x * self.std + self.mean