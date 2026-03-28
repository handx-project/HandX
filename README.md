<p align="center">
<h1><strong>[CVPR 2026] HandX: Scaling Bimanual Motion and Interaction Generation</strong></h1>
  <p align="center">
    <a href='' target='_blank'>Zimu Zhang</a><sup>1*</sup>&emsp;
    <a href='https://nboierzyc.github.io' target='_blank'>Yucheng Zhang</a><sup>1*</sup>&emsp;
    <a href='https://xiyan-xu.github.io' target='_blank'>Xiyan Xu</a><sup>1</sup>&emsp;
    <a href='https://github.com/wzyabcas' target='_blank'>Ziyin Wang</a><sup>1</sup>&emsp;
    <a href='https://sirui-xu.github.io' target='_blank'>Sirui Xu</a><sup>1&dagger;</sup>&emsp;
    <a href='' target='_blank'>Kai Zhou</a><sup>2,3</sup>&emsp;
    <br>
    <a href='' target='_blank'>Bing Zhou</a><sup>3</sup>&emsp;
    <a href='' target='_blank'>Chuan Guo</a><sup>3</sup>&emsp;
    <a href='' target='_blank'>Jian Wang</a><sup>3</sup>&emsp;
    <a href='https://yxw.web.illinois.edu/' target='_blank'>Yu-Xiong Wang</a><sup>1</sup>&emsp;
    <a href='https://lgui.web.illinois.edu/' target='_blank'>Liang-Yan Gui</a><sup>1</sup>&emsp;
    <br>
    <sup>1</sup>University of Illinois Urbana-Champaign&emsp;
    <sup>2</sup>Specs Inc.&emsp;
    <sup>3</sup>Snap Inc.
    <br>
    <sup>*</sup>Equal Contribution&emsp;
    <sup>&dagger;</sup>Project Lead&emsp;
  </p>
</p>

<p align="center">
  <a href='https://handx-project.github.io'>
    <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green'></a>
</p>

<p align="center">
  <img src="assets/teaser.png" width="100%">
</p>

## Demo

<p align="center">
  <video src="https://github.com/user-attachments/assets/d84a0405-0eda-43d5-8f93-0a2928e430f7" autoplay loop muted playsinline width="100%"></video>
</p>


## Environment Setup

<details>
  <summary>Installation steps</summary>

1. Create a conda environment:

```bash
conda create -n HandX python=3.11 -y
conda activate HandX
```

2. Install PyTorch 2.6.0 (CUDA 12.4):

```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Install PyTorch3D:

```bash
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
```
</details>

## Dataset Preparation

Place the MANO model files `MANO_LEFT.pkl` and `MANO_RIGHT.pkl` into the `diffusion/body_models/mano/` directory.

```
diffusion/
└── body_models/
    └── mano/
        ├── MANO_LEFT.pkl
        └── MANO_RIGHT.pkl
```

Download the HandX sample data archive from this [link](https://drive.google.com/file/d/1Nd2eWBwSljMuImlN9T20bX7vMUp-T59v/view?usp=sharing), all base data will be released shortly [here](https://drive.google.com/file/d/1Vc9z71RKAgic-ih1lNm2hXgB-LmL6onW/view?usp=sharing) after legal review.
This archive contains data from all datasets **except** ARCTIC and H2O, which must be obtained separately due to their redistribution policies.

Extract the archive and place the included 4 files under `data/handx/`:

```
data/
└── handx/
    ├── train_can_pos_all_wotextfeat.npz
    ├── train_mano.npz
    ├── test_can_pos_all_wotextfeat.npz
    └── test_mano.npz
```

### Processing ARCTIC and H2O

The `data/processing/` directory contains scripts to process ARCTIC and H2O data and merge them into the base dataset.

<details>
  <summary>Directory structure</summary>

```
data/processing/
├── H2O/
│   ├── raw/
│   │   ├── subject1/
│   │   ├── subject2/
│   │   ├── subject3/
│   │   └── subject4/
│   ├── skeleton/                  # step 1 output
│   ├── skeleton_canonicalized/    # step 2 output
│   ├── skeleton_split/            # step 3 output
│   ├── text/
│   └── mano/                      # step 4 output
└── ARCTIC/
    ├── raw_seqs/
    │   ├── s01/
    │   ├── s02/
    │   ├── s04/
    │   ├── s05/ ... s10/
    ├── temp/                      # step 1 output (intermediate)
    ├── skeleton/                  # step 2 output
    ├── skeleton_canonicalized/    # step 3 output
    ├── skeleton_split/            # step 4 output
    ├── text/
    └── mano/                      # step 5 output
```
</details>

<details>
  <summary>H2O</summary>

1. Download the H2O dataset from https://h2odataset.ethz.ch. You need the following files:
   - `subject1_pose_v1_1.tar`
   - `subject2_pose_v1_1.tar`
   - `subject3_pose_v1_1.tar`
   - `subject4_pose_v1_1.tar`

   Unzip them and place the contents under `data/processing/H2O/raw/`

2. Download the annotation texts from this [link](https://drive.google.com/file/d/1on_puoYsoYxVXytKW2HWP1mOzg5_FXge/view?usp=sharing) and place the Json files under `data/processing/H2O/text/`

3. Process the data by running the following commands in order:

```bash
cd data/processing/H2O

# Step 1: Extract skeleton from raw H2O data
python extract_h2o_data.py

# Step 2: Canonicalize skeleton orientation
python canonicalize_pose.py

# Step 3: Split full sequences into 60-frame windows
python split_skeleton.py

# Step 4: Fit MANO parameters to skeleton windows (requires GPU)
cd ..
python skeleton2mano.py --input_dir H2O/skeleton_split --output_dir H2O/mano
```
</details>

<details>
  <summary>ARCTIC</summary>

1. Download the ARCTIC dataset from https://arctic.is.tue.mpg.de. You need the raw_seqs files.

   Place them under `data/processing/ARCTIC/raw_seqs/`.

2. Download the annotation texts from this [link](https://drive.google.com/file/d/1KexkVATyr50BK4FVZzJy2vbRE6StjmvO/view?usp=sharing) and place the Json files under `data/processing/ARCTIC/text/`

3. Process the data by running the following commands in order:

```bash
cd data/processing/ARCTIC

# Step 1: Extract MANO parameters from raw ARCTIC data
python mano_extract.py

# Step 2: Convert MANO parameters to skeleton
python mano_to_skeleton.py

# Step 3: Canonicalize skeleton orientation
python canonicalize_pose.py

# Step 4: Split full sequences into 60-frame windows
python split_skeleton.py

# Step 5: Fit MANO parameters to skeleton windows (requires GPU)
cd ..
python skeleton2mano.py --input_dir ARCTIC/skeleton_split --output_dir ARCTIC/mano
```
</details>

<details>
  <summary>Merging</summary>

After processing both datasets, merge them into the base dataset:

```bash
cd data/processing
python merge_arctic_h2o.py
```

This appends the ARCTIC and H2O data to the existing `.npz` files in `data/handx/`, producing the complete dataset.
</details>

<details>
  <summary>(Optional) Dataset Contact Quality Evaluation</summary>

The script `compute_contact_metric.py` evaluates contact quality metrics for bimanual skeleton data, which is used in to evaluate the data quality in our article.

It computes three core metrics:
- **Contact Ratio**: Proportion of frames with hand-hand contact
- **Avg Contact Duration**: Mean length of contact segments (seconds)
- **Contact Frequency**: Number of contact events per second

```bash
cd data
python scripts/evaluation/compute_contact_metric.py /path/to/motion.npy [fps]
```

The input `.npy` file should have shape `(T, 2, 21, 3)` where T is the number of frames, 2 represents left/right hands, 21 is the number of joints per hand, and 3 is the xyz coordinates. It can be extracted from the dataset npz files.
</details>

### Converting to Autoregressive Representation

The autoregressive model uses a 288-dim motion representation. To convert the HandX dataset (`data/handx/`) into this format:

```bash
cd data/processing
python convert_to_autoregressive.py
```

This reads `data/handx/{train,test}_can_pos_all_wotextfeat.npz` and `{train,test}_mano.npz`, and outputs to `autoregressive/data/`:

```
autoregressive/data/
├── train_full_correct_duet_scalar_rot.npz
├── test_full_correct_duet_scalar_rot.npz
├── texts_all.pkl
├── mean_correct_duet_scalar_rot.npy
└── std_correct_duet_scalar_rot.npy
```

To write to a custom directory, use `--output_dir`:

```bash
python convert_to_autoregressive.py --output_dir /path/to/output
```

## Diffusion

All training and evaluation commands below should be run from the `diffusion/` directory:

```bash
cd diffusion
```

### Training

```bash
sh scripts/diffusion/train/train.sh
```

This repository uses [Hydra](https://hydra.cc/docs/intro/) to manage experiment configurations. You can modify the configuration files located in the `conf/` folder.

### Generation

#### Generating Samples

The following scripts are available for versatile generation tasks:

| **Script**                | **Task**                                     |
| ------------------------- | -------------------------------------------- |
| `run_text2motion.py`      | Text-to-motion generation (unconstrained)    |
| `run_fix_lefthand.py`     | Fix left hand, generate right hand           |
| `run_wrist_traj.py`       | Fix wrist trajectory, generate hand motion   |
| `run_inbetweening.py`     | Motion in-betweening (fix first/last frames) |
| `run_contact_keyframe.py` | Generation conditioned on contact keyframes  |
| `run_two_stage.py`        | Long-horizon generation                      |


<details>
  <summary>Option 1: Edit Script Configuration (for the first 4 scripts)</summary>

Open the corresponding script and update the `CHECKPOINTS` configuration:

```python
CHECKPOINTS = [
    {
        'name': 'your_checkpoint_name',
        'checkpoint_dir': '/path/to/your/checkpoint',
        'model_name': 'model000220000.pt',
        'num_val_samples': 256,
        'data_dir': '/path/to/your/data',
        'data_loader': 'src.diffusion.data_loader.handx.HandXDataset',
        'data_file_name': 'can_pos_all_wotextfeat.npz',
        'eval_folder_name': 'generate_xxx',
        'num_generated': 4,
        'description': 'Your description'
    }
]
```

Then run:

```bash
python scripts/evaluation/run_text2motion.py
```
</details>

<details>
  <summary>Option 2: Command Line Arguments (for the last 2 scripts)</summary>

```bash
python scripts/evaluation/run_contact_keyframe.py \
    --checkpoint_dir /path/to/your/checkpoint \
    --model_name model000070000.pt \
    --data_dir /path/to/your/data \
    --num_val_samples 256

python scripts/evaluation/run_two_stage.py \
    --checkpoint_dir /path/to/your/checkpoint \
    --model_name model000070000.pt \
    --data_dir /path/to/your/data \
    --num_val_samples 256
```
</details>

## Autoregressive

All training and evaluation commands below should be run from the `autoregressive/` directory:

```bash
cd autoregressive
```

### Dependencies

Run the script to download dependencies materials:

```bash
bash prepare/download_glove.sh
```

### Training

#### 1. Train Tokenizer

```bash
bash scripts/train/train_tokenizer.sh
```

If you don't want to use wavelet transformation, simply delete `--use_patcher`, `--patch_size` and `--patch_method` arguments.

**Codebook size** is controlled by `--nb-code`. Modify it in `train_tokenizer.sh`:

```bash
--nb-code 4096    # default, can be changed to 1024, 8192, 65536, etc.
```

#### 2. Train Text-to-Motion Model

First, run the following command to inference all of the motion codes by the trained tokenizer. Change the `--resume-pth` argument to the path of your tokenizer checkpoint.

```bash
bash scripts/train/train_t2m_get_codes.sh
```

Then train the text-to-motion model:

```bash
bash scripts/train/train_t2m_4096.sh
```

**Model size** is controlled by `--pretrained_llama` in `train_t2m_4096.sh`. Available sizes:

| Name | Layers | Heads | Embed Dim |
|------|--------|-------|-----------|
| 44M  | 8      | 8     | 512       |
| 111M | 12     | 12    | 768       |
| 222M | 16     | 16    | 1024      |
| 343M | 24     | 16    | 1024      |
| 775M | 36     | 20    | 1280      |
| 1B   | 48     | 24    | 1536      |
| 3B   | 24     | 32    | 3200      |

Example: to use a smaller 44M model, change the flag in the script:

```bash
--pretrained_llama 44M
```

Note: when changing `--nb-code` (codebook size), make sure the same value is used consistently across the tokenizer training, `get_codes`, and `train_t2m` scripts.

### Generation

To evaluate the text-to-motion sample:

```bash
bash scripts/eval/generate_for_eval.sh 
```

## Evaluation

Both diffusion and autoregressive models generate per-sample PKL files in the same format. A unified evaluation script computes all metrics on these files.

### Prerequisites

Download the evaluation encoder checkpoints from this [link](https://drive.google.com/file/d/1zR0C7vA6H991cO-TB1qQDo7odzYB1xEE/view) and place them under `evaluation/checkpoints/`:

```
evaluation/
└── checkpoints/
    ├── epoch=269.ckpt
    ├── mean_can_pos.npy
    └── std_can_pos.npy
```

Optionally, you can train your own evaluation encoder:

```bash
cd evaluation
sh train_tma.sh
```

### Running Evaluation

```bash
cd evaluation
python run_evaluation.py --output_dir /path/to/pkl/files
```

For diffusion, point `--output_dir` to the directory containing generated `val_sample_*.pkl` files. For autoregressive, point it to the output of `generate_for_eval.sh`.

<details>
  <summary>Metrics and options</summary>

Evaluation metrics:
- **FID**: Frechet Inception Distance (distribution similarity)
- **R-precision**: Text-motion matching accuracy (Top-1, Top-2, Top-3)
- **Matching Score**: Text-motion embedding distance
- **MPJPE**: Mean Per-Joint Position Error (mm)
- **Diversity**: Variation across generated samples
- **Multimodality**: Variation across different generations of same text (diffusion only, since autoregressive is deterministic)
- **Interaction**: Intra/Inter hand contact precision, recall, F1

Options:

```bash
python run_evaluation.py \
    --output_dir /path/to/pkl/files \
    --batch_size 32 \
    --delete_pkl          # delete PKL files after loading to save disk space
    --results_file eval.json  # custom output filename
```

Results are saved as `evaluation_results.json` in the current directory.
</details>

## Citation

If you find this repository useful for your work, please cite:

```bibtex
@inproceedings{zhang2026handx,
    title     = {HandX: Scaling Bimanual Motion and Interaction Generation},
    author    = {Zhang, Zimu and Zhang, Yucheng and Xu, Xiyan and Wang, Ziyin and Xu, Sirui and Zhou, Kai and Zhou, Bing and Guo, Chuan and Wang, Jian and Wang, Yu-Xiong and Gui, Liang-Yan},
    booktitle = {CVPR},
    year      = {2026},
}
```

Please also consider citing the datasets used in this work:
```bibtex
@inproceedings{fu2025gigahands,
    title     = {{GigaHands}: A Massive Annotated Dataset of Bimanual Hand Activities},
    author    = {Fu, Rao and Zhang, Dingxi and Jiang, Alex and Fu, Wanjia and Funk, Austin and Ritchie, Daniel and Sridhar, Srinath},
    booktitle = {CVPR},
    year      = {2025},
}

@inproceedings{banerjee2025hot3d,
    title     = {{HOT3D}: Hand and Object Tracking in {3D} from Egocentric Multi-View Videos},
    author    = {Banerjee, Prithviraj and Shkodrani, Sindi and Moulon, Pierre and Hampali, Shreyas and Han, Shangchen and Zhang, Fan and Zhang, Linguang and Fountain, Jade and Miller, Edward and Basol, Selen and others},
    booktitle = {CVPR},
    year      = {2025},
}

@inproceedings{fan2023arctic,
    title     = {{ARCTIC}: A Dataset for Dexterous Bimanual Hand-Object Manipulation},
    author    = {Fan, Zicong and Taheri, Omid and Tzionas, Dimitrios and Kocabas, Muhammed and Kaufmann, Manuel and Black, Michael J. and Hilliges, Otmar},
    booktitle = {CVPR},
    year      = {2023},
}

@inproceedings{kwon2021h2o,
    title     = {{H2O}: Two Hands Manipulating Objects for First Person Interaction Recognition},
    author    = {Kwon, Taein and Tekin, Bugra and St{\"u}hmer, Jan and Bogo, Federica and Pollefeys, Marc},
    booktitle = {ICCV},
    year      = {2021},
}

@inproceedings{wang2023holoassist,
    title     = {{HoloAssist}: An Egocentric Human Interaction Dataset for Interactive {AI} Assistants in the Real World},
    author    = {Wang, Xin and Kwon, Taein and Rad, Mahdi and Pan, Bowen and Chakraborty, Ishani and Andrist, Sean and Bohus, Dan and Feniello, Ashley and Tekin, Bugra and Frujeri, Felipe Vieira and others},
    booktitle = {ICCV},
    year      = {2023},
}
```
