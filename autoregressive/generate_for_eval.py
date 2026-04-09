"""
Generate motions from autoregressive model for evaluation.

Outputs (T, 2, 21, 3) joint positions in the same format as diffusion,
so the same evaluation code can be used for both.

Usage:
    python generate_for_eval.py \
        --resume-pth results/output/FSQ_96len/FSQ_4096_288_patch_haar/net_latest.pth \
        --resume-trans results/output/T2M/coodbook4096_P111M/net_last.pth \
        --num-samples 2048 \
        --out-dir results/eval_output
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
from tqdm import tqdm
from transformers import T5EncoderModel, T5TokenizerFast

import models.vqvae as vqvae
from models.lit_llama.model_hf import LLaMAHF, LLaMAHFConfig
from utils.motion_process import recover_from_local_position

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def parse_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--resume-pth', type=str, required=True, help='VQ-VAE checkpoint')
    parser.add_argument('--resume-trans', type=str, required=True, help='Transformer checkpoint')
    parser.add_argument('--pretrained_llama', type=str, default='111M')
    parser.add_argument('--nb-code', type=int, default=4096)
    parser.add_argument('--code-dim', type=int, default=512)
    parser.add_argument('--output-emb-width', type=int, default=512)
    parser.add_argument('--down-t', type=int, default=1)
    parser.add_argument('--stride-t', type=int, default=2)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--dilation-growth-rate', type=int, default=3)
    parser.add_argument('--vq-act', type=str, default='relu')
    parser.add_argument('--vq-norm', type=str, default='LN')
    parser.add_argument('--quantizer', type=str, default='FSQ')
    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--use_patcher', action='store_true')
    parser.add_argument('--patch_size', type=int, default=1)
    parser.add_argument('--patch_method', type=str, default='haar')
    parser.add_argument('--use_attn', type=bool, default=False)
    parser.add_argument('--input-dim', type=int, default=288)
    parser.add_argument('--block-size', type=int, default=351)
    parser.add_argument('--text_encode', type=str, default='flan-t5-xl')
    parser.add_argument('--text_sum_way', type=str, default='cls')
    parser.add_argument('--tie_weights', action='store_true')
    # Generation
    parser.add_argument('--num-samples', type=int, default=2048)
    parser.add_argument('--out-dir', type=str, default='results/eval_output')
    parser.add_argument('--split', type=str, default='test')
    return parser.parse_args()


def load_test_data(split='test'):
    """Load test data from npz, return list of (motion_288, text_caption) tuples."""
    npz_path = os.path.join(_DATA_DIR, f'{split}_full_correct_duet_scalar_rot.npz')
    mean = np.load(os.path.join(_DATA_DIR, 'mean_correct_duet_scalar_rot.npy'))
    std = np.load(os.path.join(_DATA_DIR, 'std_correct_duet_scalar_rot.npy')) + 1e-6

    raw_data = dict(np.load(npz_path, allow_pickle=True))

    samples = []
    for name, val in raw_data.items():
        entry = val.item()
        motion = entry['motion']  # (T, 288)
        l_texts = entry.get('left_annotation', [])
        r_texts = entry.get('right_annotation', [])
        i_texts = entry.get('interaction_annotation', [])

        if not (l_texts and r_texts and i_texts):
            continue

        # Use the last (most detailed) annotation
        lt = l_texts[-1]
        rt = r_texts[-1]
        it = i_texts[-1]
        caption = f"<extra_id_0> {lt} <extra_id_1> {rt} <extra_id_2> {it}"

        samples.append({
            'name': name,
            'motion': motion,
            'caption': caption,
        })

    return samples, mean, std


def motion_288_to_joints(motion_288, mean, std):
    """Convert normalized 288-dim motion to (T, 2, 21, 3) joint positions."""
    # Denormalize
    motion_denorm = motion_288 * std + mean
    # Convert to joint positions: (T, 2, 21, 3)
    joints = recover_from_local_position(motion_denorm, njoint=20)
    return joints


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda')

    # ---- Load test data ----
    print("Loading test data...")
    test_samples, mean, std = load_test_data(args.split)
    print(f"Loaded {len(test_samples)} test samples")

    num_samples = min(args.num_samples, len(test_samples))
    # Shuffle and select
    np.random.seed(42)
    indices = np.random.permutation(len(test_samples))[:num_samples]
    test_samples = [test_samples[i] for i in indices]
    print(f"Using {num_samples} samples for evaluation")

    # ---- Load text encoder ----
    print("Loading text encoder...")
    tokenizer = T5TokenizerFast.from_pretrained("t5-base")
    text_encoder = T5EncoderModel.from_pretrained("t5-base")
    text_encoder.set_attn_implementation("sdpa")
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.to(device)
    clip_dim = 768

    # ---- Load VQ-VAE ----
    print("Loading VQ-VAE...")
    net = vqvae.HumanVQVAE(
        args, args.nb_code, args.code_dim, args.output_emb_width,
        args.down_t, args.stride_t, args.width, args.depth,
        args.dilation_growth_rate, args.vq_act, args.vq_norm,
        args.kernel_size, args.use_patcher, args.patch_size,
        args.patch_method, args.use_attn
    )
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    if 'net' in ckpt:
        ckpt = ckpt['net']
    else:
        ckpt = ckpt.get('trans', ckpt)
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    net.load_state_dict(ckpt, strict=True)
    net.eval()
    net.to(device)
    print("VQ-VAE loaded")

    # Get actual codebook size from model
    actual_nb_code = net.vqvae.quantizer.codebook_size

    # ---- Load Transformer ----
    print("Loading Transformer...")
    config = LLaMAHFConfig.from_name(args.pretrained_llama)
    config.block_size = args.block_size
    config.vocab_size = actual_nb_code + 2
    config.clip_dim = clip_dim
    config.tie_weights = args.tie_weights

    trans = LLaMAHF(config)
    trans_ckpt = torch.load(args.resume_trans, map_location='cpu')
    trans_ckpt = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in trans_ckpt['trans'].items()}
    trans.load_state_dict(trans_ckpt, strict=True)
    trans.eval()
    trans.to(device)
    print(f"Transformer loaded ({args.pretrained_llama})")

    # ---- Generate ----
    print(f"\nGenerating {num_samples} motions...")
    gt_joints_list = []
    pred_joints_list = []
    texts_list = []

    with torch.no_grad():
        for sample in tqdm(test_samples, desc="Generating"):
            caption = sample['caption']
            gt_motion = sample['motion']  # (T, 288)

            # Convert GT to joint positions
            gt_joints = recover_from_local_position(gt_motion, njoint=20)  # (T, 2, 21, 3)

            # Encode text
            cap_inputs = tokenizer(caption, padding=True, truncation=True, return_tensors="pt")
            y_mask = cap_inputs.attention_mask.to(device)
            feat_text = text_encoder(
                input_ids=cap_inputs.input_ids.to(device),
                attention_mask=y_mask,
                output_hidden_states=False
            ).last_hidden_state  # (1, seq_len, 768)

            if args.text_sum_way == 'cls':
                feat_text = feat_text[:, 0:1, :]  # (1, 1, 768)
                y_mask_for_sample = torch.ones(1, 1, device=device)
            elif args.text_sum_way == 'mean':
                feat_text = (feat_text * y_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / y_mask.sum(dim=1, keepdim=True).unsqueeze(-1)
                y_mask_for_sample = torch.ones(1, 1, device=device)

            # Generate motion tokens
            index_motion = trans.sample(feat_text, y_mask_for_sample, False)

            # Decode to motion
            pred_pose = net.forward_decoder(index_motion)  # (1, T', 288)
            pred_pose_np = pred_pose.detach().cpu().numpy().squeeze(0)  # (T', 288)

            # Convert pred to joint positions
            pred_joints = recover_from_local_position(pred_pose_np, njoint=20)  # (T', 2, 21, 3)

            gt_joints_list.append(gt_joints)
            pred_joints_list.append(pred_joints)
            texts_list.append(caption)

    # ---- Save per-sample PKL files (matching diffusion format) ----
    # Diffusion format: gt_motion_real (60, 168), generated_real (num_generated, 60, 168)
    # 168 = 2 * 21 * 4, where 4th dim is padded with 0 (dropped by eval as [:,:,:,:3])
    fixed_len = 60

    def joints_to_168(joints, target_len=60):
        """Convert (T, 42, 3) or (T, 2, 21, 3) to (target_len, 168) with zero-padding."""
        if joints.ndim == 3:
            # (T, 42, 3) -> (T, 2, 21, 3)
            joints = joints.reshape(joints.shape[0], 2, 21, 3)
        T = joints.shape[0]
        if T >= target_len:
            joints = joints[:target_len]
        else:
            joints = np.concatenate([joints, np.zeros((target_len - T, 2, 21, 3))], axis=0)
        # Pad 4th coordinate with zeros: (60, 2, 21, 3) -> (60, 2, 21, 4)
        padded = np.concatenate([joints, np.zeros((*joints.shape[:-1], 1))], axis=-1)
        return padded.reshape(target_len, -1)  # (60, 168)

    print(f"\nSaving {num_samples} samples...")
    for i, (gt, pred, text, sample) in enumerate(tqdm(
            zip(gt_joints_list, pred_joints_list, texts_list, test_samples),
            total=num_samples, desc="Saving")):

        gt_168 = joints_to_168(gt)          # (60, 168)
        pred_168 = joints_to_168(pred)      # (60, 168)
        generated_168 = pred_168[np.newaxis]  # (1, 60, 168) — 1 generation for AR

        # Parse caption back to text_prompt dict
        # Caption format: "<extra_id_0> left <extra_id_1> right <extra_id_2> interaction"
        parts = text.split('<extra_id_1>')
        left_part = parts[0].replace('<extra_id_0>', '').strip()
        rest = parts[1].split('<extra_id_2>')
        right_part = rest[0].strip()
        interaction_part = rest[1].strip() if len(rest) > 1 else ''

        pkl_data = {
            'text_prompt': {
                'left': left_part,
                'right': right_part,
                'two_hands_relation': interaction_part,
            },
            'gt_motion_real': gt_168,            # (60, 168)
            'generated_real': generated_168,     # (1, 60, 168)
        }

        pkl_path = os.path.join(args.out_dir, f'val_sample_{i:03d}_idx{i}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(pkl_data, f)

    print(f"\nSaved {num_samples} PKL files to {args.out_dir}/")
    print("Done!")


if __name__ == '__main__':
    main()
