import json
import os
from copy import deepcopy
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from src.base import get_logger
from src.base.data_engine import (
    build_fold_dataloaders,
    build_full_dataset_arrays,
    make_kfold_indices,
)
from src.utils.experiment_utils import (
    expand_hyper_parameters,
    get_device,
    save_summaries_to_excel,
    set_seed,
    summarize_fold_results,
    train_final_model_on_full_data,
    train_one_fold,
)


@hydra.main(
    version_base="1.3",
    config_path=os.path.join(os.getcwd(), "configs"),
    config_name="source",
)
def main(cfg) -> None:
    set_seed(int(cfg.seed))

    model_name = cfg.model.name
    circuit_name = cfg.dataset.circuit_name

    result_root = Path(cfg.result_root)
    weight_root = Path(cfg.weight_root)
    log_root = Path(cfg.log_root)

    result_dir = result_root / model_name / circuit_name / "source"
    weight_dir = weight_root / model_name / circuit_name / "source"
    log_dir = log_root / model_name / circuit_name / "source"

    result_dir.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(
        log_file_path=str(log_dir / "source_experiment.log"),
        logger_name=f"{model_name}_{circuit_name}_source",
    )

    device = get_device()

    logger.info(f"Using device: {device}")
    logger.info("Loaded config:")
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    full_data_dict = build_full_dataset_arrays(
        cfg=cfg,
        logger=logger,
    )

    x_device = full_data_dict["x_device"]
    y_logged = full_data_dict["y_logged"]

    input_shape = (
        int(x_device.shape[1]),
        int(x_device.shape[2]),
    )

    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Target shape: {y_logged.shape}")

    all_hypers = expand_hyper_parameters(cfg)
    logger.info(f"Total hyperparameter groups: {len(all_hypers)}")

    fold_indices = make_kfold_indices(
        num_samples=len(x_device),
        k_fold=int(cfg.exp.k_fold),
        seed=int(cfg.seed),
    )

    summaries = []
    best_summary = None
    best_hyper = None
    best_mean_val_r2 = -1e18

    for hyper_idx, hyper in enumerate(all_hypers):
        logger.info(
            f"[Hyper {hyper_idx + 1}/{len(all_hypers)}] "
            f"{json.dumps(hyper, ensure_ascii=False)}"
        )

        batch_size = int(hyper.get("batch_size", cfg.dataset.batch_size))
        fold_results = []

        for fold_idx, (train_indices, valid_indices) in enumerate(fold_indices):
            logger.info(
                f"Hyper {hyper_idx + 1}/{len(all_hypers)} | "
                f"Fold {fold_idx + 1}/{cfg.exp.k_fold}"
            )

            fold_loader_dict = build_fold_dataloaders(
                x_device=x_device,
                y_logged=y_logged,
                train_indices=train_indices,
                valid_indices=valid_indices,
                batch_size=batch_size,
                num_workers=cfg.dataset.num_workers,
                normalize_x=cfg.dataset.normalize_x,
                normalize_y=cfg.dataset.normalize_y,
                drop_last=False,
            )

            fold_result = train_one_fold(
                cfg=cfg,
                hyper=hyper,
                fold_idx=fold_idx,
                fold_loader_dict=fold_loader_dict,
                input_shape=input_shape,
                device_messages=full_data_dict["device_messages"],
                device_level_attn_mask=full_data_dict.get("device_level_attn_mask", None),
                log_y_indices=full_data_dict["log_y_indices"],
                device=device,
                logger=logger,
            )

            fold_results.append(fold_result)

        summary = summarize_fold_results(
            hyper=hyper,
            fold_results=fold_results,
        )

        summaries.append(summary)

        logger.info(
            f"Hyper {hyper_idx + 1} | "
            f"mean_train_r2={summary['mean_train_r2']:.6f} | "
            f"mean_val_r2={summary['mean_val_r2']:.6f}"
        )

        save_summaries_to_excel(
            summaries=summaries,
            save_path=str(result_dir / "result.xlsx"),
        )

        if summary["mean_val_r2"] > best_mean_val_r2:
            best_mean_val_r2 = summary["mean_val_r2"]
            best_hyper = deepcopy(hyper)
            best_summary = deepcopy(summary)

            with open(result_dir / "hyper_parameters.json", "w", encoding="utf-8") as f:
                json.dump(best_hyper, f, indent=4, ensure_ascii=False)

            with open(result_dir / "best_cv_result.json", "w", encoding="utf-8") as f:
                json.dump(best_summary, f, indent=4, ensure_ascii=False)

            logger.info(
                f"New best hyperparameters found. "
                f"mean_val_r2={best_mean_val_r2:.6f}"
            )

    if best_hyper is None:
        raise RuntimeError("No valid hyperparameter group was evaluated.")

    logger.info(f"Best hyperparameters: {best_hyper}")

    final_model, final_loader_dict, final_train_curves = train_final_model_on_full_data(
        cfg=cfg,
        best_hyper=best_hyper,
        full_data_dict=full_data_dict,
        input_shape=input_shape,
        device=device,
        logger=logger,
    )

    save_obj = {
        "model_state_dict": final_model.state_dict(),
        "best_hyper_parameters": best_hyper,
        "best_cv_result": best_summary,

        "x_scaler": final_loader_dict["x_scaler"],
        "y_scaler": final_loader_dict["y_scaler"],

        "log_y_indices": full_data_dict["log_y_indices"],
        "log_y_columns": full_data_dict["log_y_columns"],

        "y_columns": full_data_dict["y_columns"],
        "device_messages": full_data_dict["device_messages"],
        "device_order": full_data_dict["device_order"],
        "device_feature_columns": full_data_dict["device_feature_columns"],
        "outlier_info": full_data_dict["outlier_info"],

        "final_train_curves": final_train_curves,
    }

    weight_path = weight_dir / f"{model_name}_{circuit_name}_source.pt"
    torch.save(save_obj, weight_path)

    logger.info(f"Final source model saved to: {weight_path}")


if __name__ == "__main__":
    main()