#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/public/home/202411094934/python_project/ACCFormer"
cd "${PROJECT_ROOT}"

# ================================
# Date tags
# ================================
# TARGET_DATE_TAG: 本次 target 实验结果保存日期
TARGET_DATE_TAG=$(date +"%Y-%m_%d")

# SOURCE_DATE_TAG: source 权重所在日期
# 如果 target 和 source 是同一天跑出来的，可以保持一致；
# 如果 source 是之前某天跑的，手动改成对应日期，例如 2026-05_09
SOURCE_DATE_TAG="${SOURCE_DATE_TAG:-${TARGET_DATE_TAG}}"

mkdir -p "hpc_logs/${TARGET_DATE_TAG}"

# ================================
# Experiment scope
# ================================
MODELS=(
  mlp
  res_mlp
)

CIRCUITS=(
  two_stage_folded_opamp
)

# Smoke test example:
# MODELS=(mlp)
# CIRCUITS=(two_stage_folded_opamp)

NUM_WORKERS=8

SOURCE_WEIGHT_ROOT="model_weight/${SOURCE_DATE_TAG}"
RESULT_ROOT="experiment_result/${TARGET_DATE_TAG}"
WEIGHT_ROOT="model_weight/${TARGET_DATE_TAG}"
LOG_ROOT="logs/${TARGET_DATE_TAG}"
HYDRA_ROOT="hydra_outputs/${TARGET_DATE_TAG}"

RUNNER="scripts/run/scut_slurm_run.sh"

# ================================
# Optional quick test overrides
# ================================
# 1 表示快速测试，0 表示正式训练
SMOKE_TEST=1

EXTRA_OVERRIDES=()

if [[ "${SMOKE_TEST}" == "1" ]]; then
  EXTRA_OVERRIDES+=(
    exp.epochs=2
    exp.log_interval=1
    scheduler.warmup_epochs=1
  )
fi

for model in "${MODELS[@]}"; do
  model_tag="${model//_/-}"

  for circuit in "${CIRCUITS[@]}"; do
    circuit_tag="${circuit//_/-}"

    SOURCE_CKPT="${PROJECT_ROOT}/${SOURCE_WEIGHT_ROOT}/${model}/${circuit}/source/${model}_${circuit}_source.pt"

    if [[ ! -f "${SOURCE_CKPT}" ]]; then
      echo "[WARNING] Source checkpoint not found, skip:"
      echo "  ${SOURCE_CKPT}"
      continue
    fi

    JOB_NAME="tgt-${model_tag}-${circuit_tag}"
    HYDRA_RUN_DIR="${HYDRA_ROOT}/${model}/${circuit}/target"

    echo "Submitting ${JOB_NAME}"
    echo "Source checkpoint: ${SOURCE_CKPT}"

    sbatch \
      -J "${JOB_NAME}" \
      -o "hpc_logs/${TARGET_DATE_TAG}/${JOB_NAME}.out" \
      -e "hpc_logs/${TARGET_DATE_TAG}/${JOB_NAME}.err" \
      "${RUNNER}" \
      python -m src.experiment.target_experiment \
        model="${model}" \
        target_init="${model}" \
        dataset.circuit_name="${circuit}" \
        dataset.mission_type=target \
        dataset.num_workers="${NUM_WORKERS}" \
        source_checkpoint_path="${SOURCE_CKPT}" \
        source_weight_root="${SOURCE_WEIGHT_ROOT}" \
        result_root="${RESULT_ROOT}" \
        weight_root="${WEIGHT_ROOT}" \
        log_root="${LOG_ROOT}" \
        hydra.run.dir="${HYDRA_RUN_DIR}" \
        "${EXTRA_OVERRIDES[@]}"

  done
done