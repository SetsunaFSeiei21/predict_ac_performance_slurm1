from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class ACDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        super().__init__()

        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        assert len(self.x) == len(self.y), (f"Feature sample num {len(self.x)} != target sample num {len(self.y)}")

    def __len__(self) -> int:
        
        return len(self.x)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        
        return {
            "device_features": self.x[idx],
            "targets": self.y[idx],
        }