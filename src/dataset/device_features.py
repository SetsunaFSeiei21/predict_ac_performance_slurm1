import ast, json, os
from typing import Any, Dict, List
import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

DEVICE_FEATURE_COLUMNS = ["W_or_res_or_cap", "L_or_zero", "m_or_zero"]

def load_device_messages(circuit_dir: str) -> List[Dict[str, Any]]:
    
    device_messages_file_path = os.path.join(circuit_dir, "device_messages.json")
    if not os.path.exists(device_messages_file_path):
        raise FileNotFoundError(f"Device messages can not find at {circuit_dir}")
    else:
        with open(device_messages_file_path, mode='r', encoding='utf-8') as f:
            return json.load(f)

def eval_expr_vectorized(expr: Any, design_df: pd.DataFrame) -> np.ndarray:
    
    num_samples = len(design_df)
    if expr is None:
        return np.zeros(num_samples, dtype=np.float64)
    expr = str(expr).strip()
    if expr == "":
        return np.zeros(num_samples, dtype=np.float64)
    if expr in design_df.columns:
        return design_df[expr].to_numpy(dtype = np.float64)
    try:
        value = float(expr)
        return np.full(num_samples, value, dtype=np.float64)
    except ValueError:
        pass
    tree = ast.parse(expr, mode='eval')
    
    def _eval_node(node) -> np.ndarray:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.Constant):
            return np.full(num_samples, float(node.value), dtype=np.float64)
        if isinstance(node, ast.Num):
            return np.full(num_samples, float(node.n), dtype=np.float64)
        if isinstance(node, ast.Name):
            name = node.id
            if name not in design_df.columns:
                raise KeyError(
                    f"Unknown design variable '{name}' in expression '{expr}'. "
                    f"Available columns: {list(design_df.columns)}"
                )
            return design_df[name].to_numpy(dtype=np.float64)
        if isinstance(node, ast.UnaryOp):
            value = _eval_node(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
            raise ValueError(f"Unsupported unary operator in expression: {expr}")
        if isinstance(node, ast.BinOp):
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
            raise ValueError(f"Unsupported binary operator in expression: {expr}")
        raise ValueError(f"Unsupported AST node {type(node)} in expression: {expr}")
    
    return _eval_node(tree)

def build_one_device_feature(device: Dict[str, Any], design_df: pd.DataFrame) -> np.ndarray:
    
    kind = device["kind"]
    params = device.get("params", {})
    num_samples = len(design_df)
    if kind in ["pmos", "nmos"]:
        w = eval_expr_vectorized(params.get("W", 0), design_df)
        l = eval_expr_vectorized(params.get("L", 0), design_df)
        m = eval_expr_vectorized(params.get("m", 1), design_df)
        # [num_samples, 3]
        one_device_feature = np.stack([w, l, m], axis=1)
    elif kind == "resistor":
        res = eval_expr_vectorized(params.get("res", 0), design_df)
        zero = np.zeros(num_samples, dtype=np.float64)
        # [num_samples, 3]
        one_device_feature = np.stack([res, zero, zero], axis=1)
    elif kind == "capacitor":
        cap = eval_expr_vectorized(params.get("cap", 0), design_df)
        zero = np.zeros(num_samples, dtype=np.float64)
        one_device_feature = np.stack([cap, zero, zero], axis=1)
    elif kind == "current_source":
        idc = eval_expr_vectorized(params.get("idc", 0), design_df)
        zero = np.zeros(num_samples, dtype=np.float64)
        one_device_feature = np.stack([idc, zero, zero], axis=1)
    else:
        raise ValueError(f"Unsupported device kind: {kind}")

    return one_device_feature[:, None, :]

def build_device_features_from_dataframe(design_df: pd.DataFrame, device_messages: List[Dict[str, Any]]) -> np.ndarray:

    device_feature_lst = []
    for device in device_messages:
        one_device_feature = build_one_device_feature(
            device=device,
            design_df=design_df,
        )
        device_feature_lst.append(one_device_feature)
    if len(device_feature_lst) == 0:
        raise ValueError("No device features were built. device_messages is empty.")
    device_features = np.concatenate(device_feature_lst, axis=1)
    
    return device_features.astype(np.float64)


def get_device_order(device_messages: List[Dict[str, Any]]) -> List[str]:
    return [device["name"] for device in device_messages]