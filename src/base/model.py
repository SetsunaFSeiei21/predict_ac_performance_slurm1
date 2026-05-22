import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, List, Any

class Device_BaseModel(nn.Module, ABC):
    
    def __init__(self, device_messages: List[Dict[str, Any]]) -> None:
        
        super().__init__()
        self.all_device_lst = [item['kind'] for item in device_messages]
        device_level_one_hot_tensors = self._get_device_level_one_hot_tensor()
        self.register_buffer("device_level_one_hot_tensors", device_level_one_hot_tensors)
        
    @abstractmethod
    def forward(self) -> None:
        
        raise NotImplementedError
    
    def _get_device_level_one_hot_tensor(self):
        
        all_device_type_lst = []
        for i in self.all_device_lst:
            if i not in all_device_type_lst:
                all_device_type_lst.append(i)
        return F.one_hot(torch.tensor([all_device_type_lst.index(i) for i in self.all_device_lst])).float()
    
    def param_num(self) -> int:
        
        return sum([param.nelement() for param in self.parameters()])