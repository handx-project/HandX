"""
Hand replay runner — play_dataset mode only, no RL framework dependency.

Usage (from simulation/):
  python intermimic/run.py \
    --task HandReplay \
    --cfg_env intermimic/data/cfg/grab_hand_grid.yaml \
    --motion_file custom_mano \
    [--save_images] [--headless] \
    [--grid_n N] [--grid_m M] [--grid_spacing S] [--grid_seed SEED]
"""

import os
import yaml
import numpy as np

# isaacgym must be imported before torch
from utils.config import get_args, parse_sim_params, set_np_formatting, set_seed
from env.tasks.hand_replay import HandReplay

import torch


def load_env_cfg(args):
    """Load only the environment YAML; inject CLI overrides."""
    with open(os.path.join(os.getcwd(), args.cfg_env), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # Override numEnvs from CLI
    if hasattr(args, 'num_envs') and args.num_envs > 0:
        cfg["env"]["numEnvs"] = args.num_envs

    # Grid play: set numEnvs = N*M and store grid params
    if hasattr(args, 'grid_n') and args.grid_n > 0 and args.grid_m > 0:
        cfg["env"]["numEnvs"]    = args.grid_n * args.grid_m
        cfg["env"]["gridN"]      = args.grid_n
        cfg["env"]["gridM"]      = args.grid_m
        cfg["env"]["gridSpacing"] = args.grid_spacing
        cfg["env"]["gridSeed"]   = args.grid_seed

    cfg["headless"]  = args.headless
    cfg["name"]      = args.task
    cfg["seed"]      = args.seed if args.seed is not None else -1
    cfg["args"]      = args
    cfg["task"]      = {"randomize": False}

    if args.motion_file:
        cfg["env"]["motion_file"] = args.motion_file

    cfg["env"]["playdataset"] = True
    cfg["env"]["saveImages"]  = args.save_images

    return cfg


def main():
    set_np_formatting()
    args = get_args()
    cfg  = load_env_cfg(args)

    set_seed(cfg["seed"])

    sim_params = parse_sim_params(args, cfg, {})

    task = HandReplay(
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        device_type=args.device,
        device_id=args.device_id,
        headless=args.headless,
    )

    max_t = int(task.max_episode_length.max().item())
    for t in range(max_t):
        task.play_dataset_step(t)


if __name__ == "__main__":
    main()
