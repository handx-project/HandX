"""
Convert MANO pkl files (from two-stage-batch.py) to .pt format
required by the hand replay system (hand_replay.py).

Output format: T x 245
  [0:3]    left wrist global position
  [3:51]   left hand 16 joints x 3 exp_map (global_orient + hand_pose)
  [51:54]  right wrist global position
  [54:102] right hand 16 joints x 3 exp_map
  [102:150] left hand 16 joints xyz (body keypoints)
  [150:198] right hand 16 joints xyz
  [198:245] unused (reserved, zeros)

Usage:
  cd simulation
  python mano_to_pt.py --input /path/to/pkl/files/ --output custom_mano/
"""

import sys
import os
import pickle
import argparse
import glob
import math
import numpy as np
import torch

KEY2MANO_DIR = os.path.join(os.path.dirname(__file__), '../key2mano')
sys.path.insert(0, os.path.abspath(KEY2MANO_DIR))
from mano2mesh import left_manomodel, right_manomodel


def expmap_to_quat(expmap):
    """expmap [T,3] → quaternion [T,4] (w,x,y,z)"""
    angle = expmap.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis  = expmap / angle
    q_w   = torch.cos(angle / 2)
    q_xyz = axis * torch.sin(angle / 2)
    return torch.cat([q_w, q_xyz], dim=-1)


def quat_to_expmap(q):
    """quaternion [T,4] (w,x,y,z) → expmap [T,3]"""
    q = q / q.norm(dim=-1, keepdim=True)
    q_w   = q[:, 0:1].clamp(-1 + 1e-7, 1 - 1e-7)
    q_xyz = q[:, 1:]
    angle = 2 * torch.acos(q_w)
    axis  = q_xyz / torch.sin(angle / 2).clamp(min=1e-8)
    return axis * angle


def quat_mul(q1, q2):
    """Multiply quaternions [T,4] (w,x,y,z)"""
    w1, x1, y1, z1 = q1[:,0], q1[:,1], q1[:,2], q1[:,3]
    w2, x2, y2, z2 = q2[:,0], q2[:,1], q2[:,2], q2[:,3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def mano_to_isaacgym(joints, global_orient_np):
    """
    Transform MANO output (Y-up, skeleton space) to IsaacGym world (Z-up).
    Position:  (x, y, z) → (x, -z, y)   [Rx(+90°)]
    Rotation:  q_new = q_c * q_old       [left-multiply, same as interact2mimic.py]

    interact2mimic.py uses: rotation_matrix_x * rotations
    where rotation_matrix_x = Rx(+pi/2).
    """
    T = joints.shape[0]

    # --- transform joint positions: (x, y, z) → (x, -z, y) ---
    joints_new = torch.zeros_like(joints)
    joints_new[:, :, 0] =  joints[:, :, 0]   # x stays
    joints_new[:, :, 1] = -joints[:, :, 2]   # new y = -old z
    joints_new[:, :, 2] =  joints[:, :, 1]   # new z = old y

    # --- rotation correction: Rx(+90°) left-multiply + upright_start right-multiply ---
    # Step 1: Rx(+90°) converts Y-up MANO → Z-up world  (same as interact2mimic.py)
    # Step 2: right-multiply by inv(R_upright_start) to match poselib robot zero-pose
    #         R_upright_start = quat(x=0.5,y=0.5,z=0.5,w=0.5) in scipy(x,y,z,w)
    #                         = quat(w=0.5,x=0.5,y=0.5,z=0.5) in (w,x,y,z)
    #         inv = conjugate = (w=0.5, x=-0.5, y=-0.5, z=-0.5) in (w,x,y,z)
    half = math.pi / 4
    q_c = torch.tensor([[math.cos(half), math.sin(half), 0.0, 0.0]],
                        dtype=torch.float32).expand(T, -1)   # Rx(+90°) in (w,x,y,z)
    q_inv_upright = torch.tensor([[0.5, -0.5, -0.5, -0.5]],
                                  dtype=torch.float32).expand(T, -1)  # inv(R_upright) in (w,x,y,z)

    go    = torch.from_numpy(global_orient_np).float()  # [T,3]
    q_old = expmap_to_quat(go)                           # [T,4]
    q_new = quat_mul(quat_mul(q_c, q_old), q_inv_upright)  # Rx(+90°)*q_old*inv(R_upright)
    new_go = quat_to_expmap(q_new).numpy()               # [T,3]

    return joints_new, new_go


def get_joints(mano_model, params, device='cpu'):
    """Run MANO forward pass and return joint positions [T, 16, 3]."""
    T = params['pose'].shape[0]
    mano_model.to(device)

    pose  = torch.from_numpy(params['pose']).float().to(device)   # [T, 48]
    shape = torch.from_numpy(params['shape']).float().to(device)  # [T, 10]
    trans = torch.from_numpy(params['trans']).float().to(device)  # [T, 3]

    with torch.no_grad():
        output = mano_model(
            global_orient=pose[:, :3],
            hand_pose=pose[:, 3:],
            betas=shape,
            transl=trans,
        )
    # joints: [T, 21, 3]; first 16 are non-tip joints
    joints = output.joints[:, :16, :].cpu()
    return joints


def convert_pkl_to_pt(pkl_path, output_path, device='cpu', z_offset=1.0):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    left_params  = data['left']
    right_params = data['right']
    T = left_params['pose'].shape[0]

    # --- joint positions via forward pass ---
    left_joints  = get_joints(left_manomodel,  left_params,  device)  # [T, 16, 3]
    right_joints = get_joints(right_manomodel, right_params, device)  # [T, 16, 3]

    # --- match Blender rendering convention: negate X and Y ---
    left_joints,  left_go  = mano_to_isaacgym(left_joints,  left_params['pose'][:, :3])
    right_joints, right_go = mano_to_isaacgym(right_joints, right_params['pose'][:, :3])

    # --- lift above ground ---
    left_joints[:, :, 2]  += z_offset
    right_joints[:, :, 2] += z_offset

    # --- build T x 245 tensor ---
    out = torch.zeros(T, 245)

    # [0:3]   left wrist global position (joint 0)
    out[:, 0:3]   = left_joints[:, 0, :]

    # [3:51]  left hand exp_map: corrected global_orient(3) + hand_pose(45)
    left_pose = torch.from_numpy(left_params['pose']).float()   # [T, 48]
    out[:, 3:6]   = torch.from_numpy(left_go).float()
    out[:, 6:51]  = left_pose[:, 3:]   # hand_pose unchanged

    # [51:54] right wrist global position
    out[:, 51:54] = right_joints[:, 0, :]

    # [54:102] right hand exp_map
    right_pose = torch.from_numpy(right_params['pose']).float()  # [T, 48]
    out[:, 54:57]  = torch.from_numpy(right_go).float()
    out[:, 57:102] = right_pose[:, 3:]  # hand_pose unchanged

    # [102:150] left hand 16 joints xyz (body keypoints)
    out[:, 102:150] = left_joints.reshape(T, 48)

    # [150:198] right hand 16 joints xyz
    out[:, 150:198] = right_joints.reshape(T, 48)

    # [198:245] unused (reserved, left as zeros)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(out, output_path)
    print(f"  saved {output_path}  shape={list(out.shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  default='mano_data/',
                        help='Directory with .pkl files')
    parser.add_argument('--output', default='custom_mano/',
                        help='Output directory for .pt files')
    parser.add_argument('--device', default='cpu',
                        help='cpu or cuda')
    parser.add_argument('--z_offset', type=float, default=1.0,
                        help='Z offset to lift hands above ground (default: 1.0m)')
    args = parser.parse_args()

    pkl_files = sorted(glob.glob(os.path.join(args.input, '*.pkl')))
    if not pkl_files:
        print(f"No .pkl files found in {args.input}")
        return

    print(f"Found {len(pkl_files)} pkl files, converting...")
    for pkl_path in pkl_files:
        out_name = os.path.splitext(os.path.basename(pkl_path))[0] + '.pt'
        out_path = os.path.join(args.output, out_name)
        print(f"Processing {os.path.basename(pkl_path)} -> {out_name}...")
        convert_pkl_to_pt(pkl_path, out_path, device=args.device, z_offset=args.z_offset)

    print(f"\nDone. {len(pkl_files)} files saved to {args.output}")


if __name__ == '__main__':
    main()
