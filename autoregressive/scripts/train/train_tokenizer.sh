conda activate HandX

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
