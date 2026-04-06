# Run grid play multiple times with different seeds and concatenate into one video
# Usage: bash scripts/data_replay_grab_grid_multi.sh [N] [M] [spacing] [motion_dir] [seed1] [seed2] ...
# Example: bash scripts/data_replay_grab_grid_multi.sh 10 10 0.25 custom_mano 42 123 456

# Setup virtual display
export DISPLAY=:99
if ! pgrep -f "Xvfb :99" > /dev/null; then
    echo "Starting Xvfb on display :99..."
    Xvfb :99 -screen 0 1920x1080x24 &
    sleep 2
fi

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
# Set ISAACGYM_PATH to your IsaacGym installation directory
export LD_LIBRARY_PATH="${ISAACGYM_PATH}/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0

GRID_N=${1:-10}
GRID_M=${2:-10}
SPACING=${3:-0.25}
MOTION_FILE=${4:-custom_mano}
shift 4
SEEDS=("$@")

# If no seeds provided, use 3 random ones
if [ ${#SEEDS[@]} -eq 0 ]; then
    SEEDS=(42 123 456)
fi

IMAGE_DIR="physhoi/data/images/example"
COMBINED_DIR="physhoi/data/images/combined"
mkdir -p $COMBINED_DIR
rm -f $COMBINED_DIR/*.png

GLOBAL_FRAME=0

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "=== Running seed=$SEED ==="

    # Clear per-run frames
    rm -f $IMAGE_DIR/*.png

    CUDA_LAUNCH_BLOCKING=1 python intermimic/run.py \
      --task HandReplay \
      --cfg_env intermimic/data/cfg/grab_hand_grid.yaml \
      --cfg_train intermimic/data/cfg/train/rlg/grab_hand.yaml \
      --test --play_dataset --save_images \
      --motion_file $MOTION_FILE \
      --grid_n ${GRID_N} --grid_m ${GRID_M} --grid_spacing ${SPACING} --grid_seed ${SEED}

    # Remove frame 0 (static init), copy rest to combined dir with global offset
    rm -f $IMAGE_DIR/rgb_env0_frame00000.png
    for f in $(ls $IMAGE_DIR/rgb_env0_frame*.png 2>/dev/null | sort); do
        cp "$f" "$COMBINED_DIR/frame$(printf '%06d' $GLOBAL_FRAME).png"
        GLOBAL_FRAME=$((GLOBAL_FRAME + 1))
    done
    echo "  Collected frames up to index $GLOBAL_FRAME"
done

# Create combined video
echo ""
echo "Creating video from $GLOBAL_FRAME total frames..."
OUTPUT="hand_grid_multi.mp4"
ffmpeg -y -framerate 30 -i $COMBINED_DIR/frame%06d.png -c:v libx264 -pix_fmt yuv420p -crf 20 $OUTPUT
echo "Video saved: $OUTPUT"
