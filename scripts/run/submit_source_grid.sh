#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/share/home/202411094934/python_project/predict_ac_performance"
cd "${PROJECT_ROOT}"

DATE_TAG=$(date +"%Y-%m_%d")

mkdir -p "hpc_logs/${DATE_TAG}"

# ================================
# Experiment scope
# ================================
MODELS=(
    # mlp
    # res_mlp
    # gat
    # gcn
    # accformer
    # accformer_no_grad
    accformer_no_grad_test
    # zerosim_device
    # zerosim_device_no_grad_test
    # zerosim_device_no_grad
    # global_encoder_wo_se
    # global_encoder_with_se
    # zerosim_device_pr_with_ge
    # zerosim_device_wo_se
    # zerosim_device_wo_se_no_grad
    # gat_w_gt
    # gcn_w_gt
    # zerosim_device_wo_se_pr_wo_gt
)

CIRCUITS=(
    5t_opamp
    # two_stage_opamp
    # two_stage_folded_opamp
)

# Smoke test:
# MODELS=(mlp)
# CIRCUITS=(two_stage_folded_opamp)

NUM_WORKERS=8

RESULT_ROOT="experiment_result/${DATE_TAG}"
WEIGHT_ROOT="model_weight/${DATE_TAG}"
LOG_ROOT="logs/${DATE_TAG}"
HYDRA_ROOT="hydra_outputs/${DATE_TAG}"

RUNNER="scripts/run/slurm_run.sh"

# ================================
# Optional quick test overrides
# ================================
# 1 表示快速测试，0 表示正式训练
SMOKE_TEST=0

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

    JOB_NAME="src-${model_tag}-${circuit_tag}"
    HYDRA_RUN_DIR="${HYDRA_ROOT}/${model}/${circuit}/source"

    echo "Submitting ${JOB_NAME}"

    sbatch \
      -J "${JOB_NAME}" \
      -o "hpc_logs/${DATE_TAG}/${JOB_NAME}.out" \
      -e "hpc_logs/${DATE_TAG}/${JOB_NAME}.err" \
      "${RUNNER}" \
      python -m src.experiment.source_experiment \
        model="${model}" \
        dataset.circuit_name="${circuit}" \
        dataset.mission_type=source \
        dataset.num_workers="${NUM_WORKERS}" \
        result_root="${RESULT_ROOT}" \
        weight_root="${WEIGHT_ROOT}" \
        log_root="${LOG_ROOT}" \
        hydra.run.dir="${HYDRA_RUN_DIR}" \
        "${EXTRA_OVERRIDES[@]}"

  done
done