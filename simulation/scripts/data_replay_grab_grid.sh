# Setup virtual display for rendering
export DISPLAY=:99
if ! pgrep -f "Xvfb :99" > /dev/null; then
    echo "Starting Xvfb on display :99..."
    Xvfb :99 -screen 0 1920x1080x24 &
    sleep 2
fi

# Clear previous frames
IMAGE_DIR="physhoi/data/images/example"
echo "Clearing previous frames in $IMAGE_DIR..."
rm -f $IMAGE_DIR/*.png

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
# Set ISAACGYM_PATH to your IsaacGym installation directory
export LD_LIBRARY_PATH="${ISAACGYM_PATH}/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0

# Grid play: show N x M hand pairs simultaneously
# Usage: bash scripts/data_replay_grab_grid.sh [N] [M] [spacing] [seed] [motion_dir]
#   seed: random seed for sequence sampling (-1 = random each run)
GRID_N=${1:-3}
GRID_M=${2:-3}
SPACING=${3:-1.5}
SEED=${4:--1}
MOTION_FILE=${5:-custom_mano}

CUDA_LAUNCH_BLOCKING=1 python intermimic/run.py \
  --task HandReplay \
  --cfg_env intermimic/data/cfg/grab_hand_grid.yaml \
  --cfg_train intermimic/data/cfg/train/rlg/grab_hand.yaml \
  --test --play_dataset --save_images \
  --motion_file $MOTION_FILE \
  --grid_n ${GRID_N} --grid_m ${GRID_M} --grid_spacing ${SPACING} --grid_seed ${SEED}

# Remove frame 0 (static init frame) and rename remaining to start from 00000
rm -f $IMAGE_DIR/rgb_env0_frame00000.png
i=0
for f in $(ls $IMAGE_DIR/rgb_env0_frame*.png | sort); do
    mv "$f" "$IMAGE_DIR/rgb_env0_frame$(printf '%05d' $i).png"
    i=$((i+1))
done

# Create video from saved frames
echo ""
echo "Creating video..."
OUTPUT="hand_grid.mp4"
ffmpeg -y -framerate 30 -i $IMAGE_DIR/rgb_env0_frame%05d.png -c:v libx264 -pix_fmt yuv420p -crf 20 $OUTPUT
echo "Video saved: $OUTPUT"
