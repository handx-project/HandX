"""
Extract MANO sequences from HandX npz files and save as pkl files
compatible with mano_to_pt.py (and the simulation pipeline).

Usage:
  # Extract specific indices
  python npz_to_pkl.py --npz ../data/handx/test_mano.npz --indices 0 5 10 --output_dir ./pkl_out

  # Randomly sample n sequences
  python npz_to_pkl.py --npz ../data/handx/train_mano.npz --random 20 --output_dir ./pkl_out

  # Both at the same time
  python npz_to_pkl.py --npz ../data/handx/test_mano.npz --indices 0 1 2 --random 5 --output_dir ./pkl_out

  # Set random seed for reproducibility
  python npz_to_pkl.py --npz ../data/handx/test_mano.npz --random 10 --seed 42 --output_dir ./pkl_out
"""

import argparse
import os
import pickle
import numpy as np


def npz_entry_to_pkl_dict(entry):
    """Convert an npz entry to the pkl format expected by mano_to_pt.py."""
    return {
        'left': {
            'shape': entry['left_shape'],
            'pose': entry['left_pose'],
            'trans': entry['left_trans'],
        },
        'right': {
            'shape': entry['right_shape'],
            'pose': entry['right_pose'],
            'trans': entry['right_trans'],
        },
    }


def save_pkl(data_dict, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(data_dict, f)


def main():
    parser = argparse.ArgumentParser(
        description='Extract MANO sequences from HandX npz and save as pkl for simulation')
    parser.add_argument('--npz', required=True,
                        help='Path to train_mano.npz or test_mano.npz')
    parser.add_argument('--indices', type=int, nargs='+', default=None,
                        help='Specific indices to extract')
    parser.add_argument('--random', type=int, default=None,
                        help='Number of random sequences to sample')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for pkl files')
    args = parser.parse_args()

    if args.indices is None and args.random is None:
        parser.error('Must specify at least one of --indices or --random')

    print(f'Loading {args.npz} ...')
    npz = np.load(args.npz, allow_pickle=True)
    all_keys = list(npz.keys())
    num_total = len(all_keys)
    print(f'  Total sequences: {num_total}')

    # Determine the split name for output naming (e.g. "test" or "train")
    basename = os.path.splitext(os.path.basename(args.npz))[0]  # "test_mano"

    os.makedirs(args.output_dir, exist_ok=True)

    indices_to_save = set()

    # 1. Specific indices
    if args.indices is not None:
        for idx in args.indices:
            if str(idx) not in all_keys:
                print(f'  Warning: index {idx} not found in npz (max key: {num_total - 1}), skipping')
            else:
                indices_to_save.add(idx)

    # 2. Random sampling
    if args.random is not None:
        rng = np.random.default_rng(args.seed)
        n = min(args.random, num_total)
        # Exclude already selected indices from the pool to avoid duplicates
        pool = [int(k) for k in all_keys if int(k) not in indices_to_save]
        n = min(n, len(pool))
        sampled = rng.choice(pool, size=n, replace=False)
        indices_to_save.update(sampled.tolist())

    indices_to_save = sorted(indices_to_save)
    print(f'  Extracting {len(indices_to_save)} sequences ...')

    for idx in indices_to_save:
        entry = npz[str(idx)].item()
        pkl_dict = npz_entry_to_pkl_dict(entry)
        out_path = os.path.join(args.output_dir, f'{basename}_{idx}.pkl')
        save_pkl(pkl_dict, out_path)

    print(f'Done. {len(indices_to_save)} pkl files saved to {args.output_dir}')


if __name__ == '__main__':
    main()
