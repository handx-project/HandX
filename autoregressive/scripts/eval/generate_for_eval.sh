#!/bin/bash
#SBATCH --account=bbsg-delta-gpu
#SBATCH --time=04:00:00
#SBATCH --partition=gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60g
#SBATCH --job-name=generate_for_eval

cd /work/hdd/bbsg/yzhang62/HANDX/autoregressive

export WANDB_MODE=disabled

python generate_for_eval.py \
--resume-pth results/output/FSQ_96len/FSQ_4096_288_patch_haar/net_latest.pth \
--resume-trans results/output/T2M/coodbook4096_P111M/net_last.pth \
--pretrained_llama 111M \
--nb-code 4096 \
--block-size 351 \
--text_encode flan-t5-xl \
--text_sum_way cls \
--down-t 1 \
--depth 3 \
--dilation-growth-rate 3 \
--vq-act relu \
--vq-norm LN \
--kernel-size 3 \
--use_patcher \
--patch_size 1 \
--patch_method haar \
--num-samples 2048 \
--out-dir results/eval_output
