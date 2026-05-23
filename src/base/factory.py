from typing import Any, Dict, Tuple, List, Optional
import torch.nn as nn
import numpy as np
from src.models import MLP, ResMLP, ACCFormer, GCN, GAT, Zerosim_Device, Zerosim_Device_WO_SE, Global_Encoder_WO_SE, Ablation_Global_Encoder_WO_CG_WO_SE,\
    Ablation_Global_Encoder_WO_GE_WO_SE, Global_Encoder_WITH_SE, Zerosim_DEVICE_PR_WITH_GE, Zerosim_Device_WO_SE_PR_WO_GT, GCN_W_GT, GAT_W_GT

def build_model(model_name: str, model_hyper_parameters: Dict[str, Any], input_shape: Tuple[int, int],
                device_messages: List[Dict[str, Any]], device_level_attn_mask: Optional[np.ndarray] = None) -> nn.Module:
    
    model_name = model_name.lower()
    device_feature_dim = input_shape[1]
    
    if model_name == "mlp":
        model = MLP(
            feature_dim=device_feature_dim,
            hidden_dim=int(model_hyper_parameters["hidden_dim"]),
            output_dim=int(model_hyper_parameters['output_dim']),
            dropout=float(model_hyper_parameters.get("dropout", 0.0)),
            embedding_layer_num=int(model_hyper_parameters.get("embedding_layer_num", 2)),
            num_heads=int(model_hyper_parameters.get("num_heads", 8)),
            decoder_layer_num=int(model_hyper_parameters.get("decoder_layer_num", 1)),
            output_layer_num=int(model_hyper_parameters['output_layer_num']),
            performance_num = int(model_hyper_parameters['performance_num']),
            device_messages=device_messages
        )
        return model
    elif model_name == 'res_mlp':
        model = ResMLP(
            feature_dim=device_feature_dim,
            hidden_dim=int(model_hyper_parameters['hidden_dim']),
            output_dim=int(model_hyper_parameters['output_dim']),
            dropout=float(model_hyper_parameters.get("dropout", 0.0)),
            res_block_num=int(model_hyper_parameters.get("res_block_num", 1)),
            num_heads=int(model_hyper_parameters.get("num_heads", 8)),
            decoder_layer_num=int(model_hyper_parameters.get("decoder_layer_num", 1)),
            output_layer_num=int(model_hyper_parameters['output_layer_num']),
            performance_num = int(model_hyper_parameters['performance_num']),
            device_messages=device_messages
        )
        return model
    elif model_name == 'accformer':
        if device_level_attn_mask is None:
            raise ValueError("ACCFormer requires device_level_attn_mask.")
        model = ACCFormer(
            feature_dim=device_feature_dim, 
            hidden_dim=int(model_hyper_parameters['hidden_dim']),
            output_dim=int(model_hyper_parameters['output_dim']),
            dropout=float(model_hyper_parameters.get("dropout", 0.0)),
            num_heads=int(model_hyper_parameters.get("num_heads", 8)),
            embedding_layer_num=int(model_hyper_parameters.get("embedding_layer_num", 2)),
            encoder_layer_num=int(model_hyper_parameters.get("encoder_layer_num", 2)),
            decoder_layer_num=int(model_hyper_parameters.get("decoder_layer_num", 1)),
            output_layer_num=int(model_hyper_parameters['output_layer_num']),
            performance_num = int(model_hyper_parameters['performance_num']),
            device_messages=device_messages,
            attn_mask=device_level_attn_mask, 
        )
        return model
    elif model_name == 'gcn':
        if device_level_attn_mask is None:
            raise ValueError("GCN requires device_level_attn_mask.")
        model = GCN(
            feature_dim=device_feature_dim,
            hidden_dim=int(model_hyper_parameters['hidden_dim']),
            output_dim=int(model_hyper_parameters['output_dim']),
            dropout=float(model_hyper_parameters.get("dropout", 0.0)),
            embedding_layer_num=int(model_hyper_parameters.get("embedding_layer_num", 2)),
            gcn_layer_num=int(model_hyper_parameters.get("gcn_layer_num", 3)),
            num_heads=int(model_hyper_parameters.get("num_heads", 8)),
            decoder_layer_num=int(model_hyper_parameters.get("decoder_layer_num", 1)),
            output_layer_num=int(model_hyper_parameters['output_layer_num']),
            performance_num = int(model_hyper_parameters['performance_num']),
            device_messages=device_messages,
            adj_mask=device_level_attn_mask, 
        )
        return model
    elif model_name == 'gat':
        if device_level_attn_mask is None:
            raise ValueError("GAT requires device_level_attn_mask.")
        model = GAT(
            feature_dim=device_feature_dim,
            hidden_dim=int(model_hyper_parameters['hidden_dim']),
            output_dim=int(model_hyper_parameters['output_dim']),
            dropout=float(model_hyper_parameters.get("dropout", 0.0)),
            embedding_layer_num=int(model_hyper_parameters.get("embedding_layer_num", 2)),
            gat_layer_num=int(model_hyper_parameters.get("gat_layer_num", 3)),
            num_heads=int(model_hyper_parameters.get("num_heads", 8)),
            decoder_layer_num=int(model_hyper_parameters.get("decoder_layer_num", 1)),
            output_layer_num=int(model_hyper_parameters['output_layer_num']),
            performance_num = int(model_hyper_parameters['performance_num']),
            device_messages=device_messages,
            adj_mask=device_level_attn_mask, 
        )
        return model
    elif model_name == 'zerosim_device':
        if device_level_attn_mask is None:
            raise ValueError("zerosim_device requires device_level_attn_mask.")
        model = Zerosim_Device(
            feature_dim=device_feature_dim,
            hidden_dim=int(model_hyper_parameters['hidden_dim']),
            output_dim=int(model_hyper_parameters['output_dim']),
            dropout=float(model_hyper_parameters.get("dropout", 0.0)),
            num_heads=int(model_hyper_parameters.get("num_heads", 8)),
            embedding_layer_num=int(model_hyper_parameters.get("embedding_layer_num", 2)),
            structure_encoding_layer_num=int(model_hyper_parameters.get("structure_encoding_layer_num", 3)),
            parameter_injection_layer_num=int(model_hyper_parameters.get("parameter_injection_layer_num", 3)),
            decoder_layer_num=int(model_hyper_parameters.get("decoder_layer_num", 1)),
            output_layer_num=int(model_hyper_parameters['output_layer_num']),
            performance_num = int(model_hyper_parameters['performance_num']),
            device_messages=device_messages,
            adj_mask=device_level_attn_mask, 
        )
        return model
    elif model_name == 'zerosim_device_wo_se':
        if device_level_attn_mask is None:
            raise ValueError("zerosim_device_wo_se requires device_level_attn_mask.")
        model = Zerosim_Device_WO_SE(
            feature_dim=device_feature_dim,
            hidden_dim=int(model_hyper_parameters['hidden_dim']),
            output_dim=int(model_hyper_parameters['output_dim']),
            dropout=float(model_hyper_parameters.get("dropout", 0.0)),
            num_heads=int(model_hyper_parameters.get("num_heads", 8)),
            embedding_layer_num=int(model_hyper_parameters.get("embedding_layer_num", 2)),
            parameter_injection_layer_num=int(model_hyper_parameters.get("parameter_injection_layer_num", 3)),
            decoder_layer_num=int(model_hyper_parameters.get("decoder_layer_num", 1)),
            output_layer_num=int(model_hyper_parameters['output_layer_num']),
            performance_num = int(model_hyper_parameters['performance_num']),
            device_messages=device_messages,
            adj_mask=device_level_attn_mask, 
        )
        return model
    elif model_name == 'global_encoder_wo_se':
        if device_level_attn_mask is None:
            raise ValueError("global_encoder_wo_se requires device_level_attn_mask.")
        model = Global_Encoder_WO_SE(
            feature_dim=device_feature_dim,
            hidden_dim=int(model_hyper_parameters['hidden_dim']),
            output_dim=int(model_hyper_parameters['output_dim']),
            dropout=float(model_hyper_parameters.get("dropout", 0.0)),
            num_heads=int(model_hyper_parameters.get("num_heads", 8)),
            embedding_layer_num=int(model_hyper_parameters.get("embedding_layer_num", 2)),
            encoder_layer_num=int(model_hyper_parameters.get("encoder_layer_num", 3)),
            decoder_layer_num=int(model_hyper_parameters.get("decoder_layer_num", 1)),
            output_layer_num=int(model_hyper_parameters['output_layer_num']),
            performance_num = int(model_hyper_parameters['performance_num']),
            unmask_global_encoder_layer_num=int(model_hyper_parameters["unmask_global_encoder_layer_num"]),
            device_messages=device_messages,
            adj_mask=device_level_attn_mask, 
        )
        return model
    elif model_name == 'ablation_ge_wo_cg_wo_se':
        if device_level_attn_mask is None:
            raise ValueError("ablation_ge_wo_cg_wo_se requires device_level_attn_mask.")
        model = Ablation_Global_Encoder_WO_CG_WO_SE(
            feature_dim=device_feature_dim,
            hidden_dim=int(model_hyper_parameters['hidden_dim']),
            output_dim=int(model_hyper_parameters['output_dim']),
            dropout=float(model_hyper_parameters.get("dropout", 0.0)),
            num_heads=int(model_hyper_parameters.get("num_heads", 8)),
            embedding_layer_num=int(model_hyper_parameters.get("embedding_layer_num", 2)),
            encoder_layer_num=int(model_hyper_parameters.get("encoder_layer_num", 3)),
            unmask_encoder_layer_num=int(model_hyper_parameters.get("unmask_encoder_layer_num", 1)),
            decoder_layer_num=int(model_hyper_parameters.get("decoder_layer_num", 1)),
            output_layer_num=int(model_hyper_parameters['output_layer_num']),
            performance_num = int(model_hyper_parameters['performance_num']),
            device_messages=device_messages,
            adj_mask=device_level_attn_mask, 
        )
        return model
    elif model_name == "ablation_ge_wo_ge_wo_se":
        model = Ablation_Global_Encoder_WO_GE_WO_SE(
            feature_dim=device_feature_dim, 
            hidden_dim=int(model_hyper_parameters['hidden_dim']),
            output_dim=int(model_hyper_parameters['output_dim']),
            dropout=float(model_hyper_parameters.get("dropout", 0.0)),
            num_heads=int(model_hyper_parameters.get("num_heads", 8)),
            embedding_layer_num=int(model_hyper_parameters.get("embedding_layer_num", 2)),
            decoder_layer_num=int(model_hyper_parameters.get("decoder_layer_num", 1)),
            output_layer_num=int(model_hyper_parameters['output_layer_num']),
            performance_num = int(model_hyper_parameters['performance_num']),
            device_messages=device_messages,
        )
        return model
    elif model_name == "global_encoder_with_se":
        if device_level_attn_mask is None:
            raise ValueError("global_encoder_with_se requires device_level_attn_mask.")
        model = Global_Encoder_WITH_SE(
            feature_dim=device_feature_dim,
            hidden_dim=int(model_hyper_parameters['hidden_dim']),
            output_dim=int(model_hyper_parameters['output_dim']),
            dropout=float(model_hyper_parameters.get("dropout", 0.0)),
            num_heads=int(model_hyper_parameters.get("num_heads", 8)),
            embedding_layer_num=int(model_hyper_parameters.get("embedding_layer_num", 2)),
            structure_encoding_layer_num=int(model_hyper_parameters.get("structure_encoding_layer_num", 1)),
            encoder_layer_num=int(model_hyper_parameters.get("encoder_layer_num", 3)),
            unmask_encoder_layer_num=int(model_hyper_parameters.get("unmask_encoder_layer_num", 1)),
            decoder_layer_num=int(model_hyper_parameters.get("decoder_layer_num", 1)),
            output_layer_num=int(model_hyper_parameters['output_layer_num']),
            performance_num = int(model_hyper_parameters['performance_num']),
            device_messages=device_messages,
            adj_mask=device_level_attn_mask, 
        )
        return model
    elif model_name == "zerosim_device_pr_with_ge":
        if device_level_attn_mask is None:
            raise ValueError("Zerosim_DEVICE_PR_WITH_GE requires device_level_attn_mask.")
        model = Zerosim_DEVICE_PR_WITH_GE(
            feature_dim=device_feature_dim,
            hidden_dim=int(model_hyper_parameters['hidden_dim']),
            output_dim=int(model_hyper_parameters['output_dim']),
            dropout=float(model_hyper_parameters.get("dropout", 0.0)),
            num_heads=int(model_hyper_parameters.get("num_heads", 8)),
            embedding_layer_num=int(model_hyper_parameters.get("embedding_layer_num", 2)),
            structure_encoding_layer_num=int(model_hyper_parameters.get("structure_encoding_layer_num", 1)),
            parameter_injection_layer_num=int(model_hyper_parameters.get("parameter_injection_layer_num", 3)),
            decoder_layer_num=int(model_hyper_parameters.get("decoder_layer_num", 1)),
            output_layer_num=int(model_hyper_parameters['output_layer_num']),
            performance_num = int(model_hyper_parameters['performance_num']),
            device_messages=device_messages,
            adj_mask=device_level_attn_mask, 
        )
        return model
    elif model_name == "zerosim_device_wo_se_pr_wo_gt":
        if device_level_attn_mask is None:
            raise ValueError("zerosim_device_wo_se_pr_wo_gt requires device_level_attn_mask.")
        model = Zerosim_Device_WO_SE_PR_WO_GT(
            feature_dim=device_feature_dim,
            hidden_dim=int(model_hyper_parameters['hidden_dim']),
            output_dim=int(model_hyper_parameters['output_dim']),
            dropout=float(model_hyper_parameters.get("dropout", 0.0)),
            num_heads=int(model_hyper_parameters.get("num_heads", 8)),
            embedding_layer_num=int(model_hyper_parameters.get("embedding_layer_num", 2)),
            parameter_injection_layer_num=int(model_hyper_parameters.get("parameter_injection_layer_num", 3)),
            decoder_layer_num=int(model_hyper_parameters.get("decoder_layer_num", 1)),
            output_layer_num=int(model_hyper_parameters['output_layer_num']),
            performance_num = int(model_hyper_parameters['performance_num']),
            device_messages=device_messages,
            adj_mask=device_level_attn_mask, 
        )
        return model
    elif model_name == "gat_w_gt":
        if device_level_attn_mask is None:
            raise ValueError("gat_w_gt requires device_level_attn_mask.")
        model = GAT_W_GT(
            feature_dim=device_feature_dim,
            hidden_dim=int(model_hyper_parameters['hidden_dim']),
            output_dim=int(model_hyper_parameters['output_dim']),
            dropout=float(model_hyper_parameters.get("dropout", 0.0)),
            num_heads=int(model_hyper_parameters.get("num_heads", 8)),
            embedding_layer_num=int(model_hyper_parameters.get("embedding_layer_num", 2)),
            gat_layer_num=int(model_hyper_parameters.get("gat_layer_num", 3)),
            decoder_layer_num=int(model_hyper_parameters.get("decoder_layer_num", 1)),
            output_layer_num=int(model_hyper_parameters['output_layer_num']),
            performance_num = int(model_hyper_parameters['performance_num']),
            device_messages=device_messages,
            adj_mask=device_level_attn_mask, 
        )
        return model
    elif model_name == "gcn_w_gt":
        if device_level_attn_mask is None:
            raise ValueError("gcn_w_gt requires device_level_attn_mask.")
        model = GCN_W_GT(
            feature_dim=device_feature_dim,
            hidden_dim=int(model_hyper_parameters['hidden_dim']),
            output_dim=int(model_hyper_parameters['output_dim']),
            dropout=float(model_hyper_parameters.get("dropout", 0.0)),
            num_heads=int(model_hyper_parameters.get("num_heads", 8)),
            embedding_layer_num=int(model_hyper_parameters.get("embedding_layer_num", 2)),
            gcn_layer_num=int(model_hyper_parameters.get("gat_layer_num", 3)),
            decoder_layer_num=int(model_hyper_parameters.get("decoder_layer_num", 1)),
            output_layer_num=int(model_hyper_parameters['output_layer_num']),
            performance_num = int(model_hyper_parameters['performance_num']),
            device_messages=device_messages,
            adj_mask=device_level_attn_mask, 
        )
        return model

    raise ValueError(f"Unsupported model name: {model_name}")

def infer_input_output_shape_from_batch(batch):
    
    device_features = batch["device_features"]
    if device_features.ndim != 3:
        raise ValueError(
            f"Expected device_features shape [B, N, F], got {device_features.shape}"
        )
        
    input_shape = (
        int(device_features.shape[1]),  # num_devices
        int(device_features.shape[2]),  # device_feature_dim
    )
    
    return input_shape