#!/bin/bash
#SBATCH --account=b_phzhwu
#SBATCH --partition=ex01A800
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=10-00:00:00
#SBATCH --mail-user=934104070@qq.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --export=NONE

set -euo pipefail

# ============================================================
# Project
# ============================================================

PROJECT_ROOT="/share/home/202411094934/python_project/predict_ac_performance"

cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ============================================================
# Clean environment
# 不依赖 ~/.bashrc
# ============================================================

export PATH="/bin:/usr/bin:/usr/local/bin:/sbin:/usr/sbin"
export LD_LIBRARY_PATH=""

# ============================================================
# CUDA
# ============================================================

export CUDA_HOME="/share/software/cuda/local/cuda-12.3"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# ============================================================
# Conda
# 这里优先使用 /share/software/anaconda3
# 如果实际 conda 在别的地方，再改这一行即可
# ============================================================

export CONDA_HOME="/share/software/anaconda3"
export PATH="${CONDA_HOME}/bin:${PATH}"

if [ -f "${CONDA_HOME}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_HOME}/etc/profile.d/conda.sh"
else
    echo "[ERROR] Cannot find conda.sh at ${CONDA_HOME}/etc/profile.d/conda.sh"
    exit 1
fi

conda activate newbase

# ============================================================
# HuggingFace mirror
# ============================================================

export HF_ENDPOINT="https://hf-mirror.com"

# ============================================================
# ICU
# ============================================================

export ICU_HOME="/share/home/202411094934/icu"
export PATH="${ICU_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${ICU_HOME}/lib:${LD_LIBRARY_PATH}"
export PKG_CONFIG_PATH="${ICU_HOME}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

# ============================================================
# PostgreSQL
# ============================================================

export PGHOME="/share/home/202411094934/pgsql17"
export PGDATA="${PGHOME}/data"
export PATH="${PGHOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${PGHOME}/lib:${LD_LIBRARY_PATH}"

# ============================================================
# Ollama
# 如果这个任务不用 ollama，可以删掉这两行
# ============================================================

export PATH="${PATH}:/share/software/ollama/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/share/software/ollama/lib"

# ============================================================
# Debug information
# ============================================================

echo "=========================================="
echo "[ACCFormer Slurm Runner]"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Job Name: ${SLURM_JOB_NAME:-N/A}"
echo "Account: ${SLURM_JOB_ACCOUNT:-N/A}"
echo "Partition: ${SLURM_JOB_PARTITION:-N/A}"
echo "Node List: ${SLURM_NODELIST:-N/A}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "Project Root: ${PROJECT_ROOT}"
echo "Working Dir: $(pwd)"
echo "Command: $@"
echo "------------------------------------------"
echo "CONDA_HOME: ${CONDA_HOME}"
echo "Conda env: ${CONDA_DEFAULT_ENV:-N/A}"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA_HOME: ${CUDA_HOME}"
echo "nvcc: $(which nvcc || true)"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo "=========================================="

# ============================================================
# Run
# ============================================================

exec "$@"