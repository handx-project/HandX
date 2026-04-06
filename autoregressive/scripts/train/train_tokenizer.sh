#!/bin/bash
#SBATCH --account=bbsg-delta-gpu
#SBATCH --time=18:00:00
#SBATCH --partition=gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --job-name=train_tokenizer

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda deactivate || true
conda deactivate || true

module purge || true
module reset || true
module unload cudatoolkit || true
module load cuda/11.8

conda activate HandX

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

export WANDB_MODE=disabled
export NCCL_TIMEOUT=1200

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 29701 train_tokenizer.py \
--batch-size 512 \
--lr 2e-4 \
--total-iter 600000 \
--lr-scheduler 300000 \
--out-dir results/output/FSQ_96len \
--exp-name FSQ_4096_288_patch_haar \
--nb-code 4096 \
--warm-up-iter 8000 \
--window-size 48 \
--loss-vel 0.5 \
--use_patcher \
--save-iter 10000 \
--save-latest 1000 \
--print-iter 200 \
--num-workers 64
