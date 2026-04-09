"""
Annotate a single .npy skeleton file using an LLM.

The .npy file should have shape (T, 2, 21, 3):
  T: number of frames
  2: two hands (left=0, right=1)
  21: joints per hand
  3: xyz coordinates
The same as our unified data format.

Usage:
  python src/annotate_single.py --npy path/to/motion.npy --output annotation.json
  python src/annotate_single.py --npy path/to/motion.npy --output annotation.json --frame_start 0 --frame_end 60
  python src/annotate_single.py --npy path/to/motion.npy --output annotation.json --model gemini-2.5-pro
"""

import sys
sys.path.append("./diffusion")

import argparse
import json
import numpy as np
from pathlib import Path

from src.generate_anno import generate_annotation
from src.feature.single_motioncode import InvalidJointDataError

with open("diffusion/src/llm/prompt.txt", "r") as f:
    prompt_template = f.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy", required=True, help="Path to input .npy file (T, 2, 21, 3)")
    parser.add_argument("--output", required=True, help="Path to output .json file")
    parser.add_argument("--frame_start", type=int, default=None, help="Start frame (default: 0)")
    parser.add_argument("--frame_end", type=int, default=None, help="End frame (default: last frame)")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro", help="LLM model to use")
    args = parser.parse_args()

    npy_path = Path(args.npy)
    if not npy_path.exists():
        print(f"ERROR: File not found: {npy_path}")
        sys.exit(1)

    motion = np.load(npy_path)  # (T, 2, 21, 3)
    print(f"Loaded {npy_path}, shape={list(motion.shape)}")

    frame_start = args.frame_start if args.frame_start is not None else 0
    frame_end = args.frame_end if args.frame_end is not None else motion.shape[0]
    motion_clip = motion[frame_start:frame_end]
    print(f"Using frames [{frame_start}:{frame_end}] → clip shape={list(motion_clip.shape)}")

    try:
        annotation = generate_annotation(
            skeleton_motion=motion_clip,
            prompt_template=prompt_template,
            model=args.model,
            return_json=True
        )
    except InvalidJointDataError as e:
        print(f"ERROR: Invalid joint data — {e}")
        sys.exit(1)

    result = {
        "seq_id": npy_path.stem,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "annotation": annotation
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Saved annotation to {output_path}")


if __name__ == "__main__":
    main()
