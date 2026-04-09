conda activate HandX
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
--num-samples 4 \
--out-dir results/eval_output
