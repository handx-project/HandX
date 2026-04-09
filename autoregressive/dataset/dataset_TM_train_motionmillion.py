import os
import random
import pickle
import codecs as cs

import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from os.path import join as pjoin
from tqdm import tqdm

import clip


##############################################
# Dataset
##############################################

class Text2MotionDataset_motionmillion(data.Dataset):
    """
    Refactored dataset:
      - NO model calls here
      - NO .to(device) here
      - Only CPU tensors & captions
    """

    def __init__(
        self,
        dataset_name,
        split,
        codebook_size,
        tokenizer_name,
        motion_type=None,
        text_type=None,
        version=None,
        unit_length=4,
        text_encode="clip",
        text_sum_way="cls",
        debug=False,
        meta_dir='./data',
    ):
        self.meta_dir = meta_dir
        self.pointer = 0
        self.dataset_name = dataset_name
        self.motion_type = motion_type
        self.text_type = text_type
        self.version = version

        self.unit_length = unit_length
        self.mot_end_idx = codebook_size          # end token id
        self.mot_pad_idx = codebook_size + 1      # pad token id

        self.tokenizer_name = tokenizer_name
        self.text_encode = text_encode
        self.text_sum_way = text_sum_way

        # ---------------- Dataset-specific config ----------------
        self.data_root = './dataset/HandX'
        self.motion_dir = pjoin(self.data_root, 'motion_data', self.motion_type)
        self.text_dir = pjoin(self.data_root, self.text_type)
        self.max_motion_length = 281
        self.max_text_length = 250

        # ---------------- Load tokenized data ----------------
        with open(os.path.join(self.data_root, self.tokenizer_name + '.pkl'), "rb") as f:
            all_data = pickle.load(f)

        # Build id_list from npz keys (auto-generate split file if missing)
        split_file = os.path.join(self.meta_dir, f'{split}_full_valid.txt')
        if not os.path.isfile(split_file):
            npz_path = os.path.join(self.meta_dir, f'{split}_full_correct_duet_scalar_rot.npz')
            npz_data = np.load(npz_path, allow_pickle=True)
            with open(split_file, 'w') as f:
                for name in sorted(npz_data.keys()):
                    f.write(name + '\n')

        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    id_list.append(line)

        if debug:
            id_list = id_list[:1000]

        data_dict = {}
        new_name_list = []

        # all_data["code_data"][name] : list/array of motion token sequences
        # all_data["text_data"][name] : dict with left/right/interaction_annotation
        for name in tqdm(id_list, desc=f"Loading {dataset_name}-{split}"):
            if name not in all_data["code_data"] or name not in all_data["text_data"]:
                continue

            code_data_ref = all_data["code_data"][name]
            text_data_ref = all_data["text_data"][name]

            if len(code_data_ref) == 0:
                continue

            data_dict[name] = {
                "m_token_list": code_data_ref,
                "text": text_data_ref,
            }
            new_name_list.append(name)

        self.data_dict = data_dict
        self.name_list = new_name_list

        print(f"[Dataset] Loaded {len(self.data_dict)} items.")

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        name = self.name_list[idx]
        entry = self.data_dict[name]
        m_token_list = entry["m_token_list"]
        text_dict = entry["text"]

        # ---- sample motion tokens ----
        m_tokens = random.choice(m_token_list)
        m_tokens = m_tokens.astype(np.int64)

        # ---- build caption from annotations ----
        l_text_list = text_dict['left_annotation'][:]
        r_text_list = text_dict['right_annotation'][:]
        i_text_list = text_dict['interaction_annotation'][:]

        ltext = random.choice(l_text_list)
        rtext = random.choice(r_text_list)
        itext = random.choice(i_text_list)

        caption = f"<extra_id_0> {ltext} <extra_id_1> {rtext} <extra_id_2> {itext}"

        return caption, m_tokens, np.array(m_tokens.shape[0])


##############################################
# DataLoader factory
##############################################

def DATALoader(
    dataset_name,
    batch_size,
    codebook_size,
    tokenizer_name,
    split,
    text_encode,
    text_sum_way,
    motion_type=None,
    text_type=None,
    version=None,
    unit_length=4,
    num_workers=4,
    debug=False,
):
    dataset = Text2MotionDataset_motionmillion(
        dataset_name=dataset_name,
        split=split,
        codebook_size=codebook_size,
        tokenizer_name=tokenizer_name,
        motion_type=motion_type,
        text_type=text_type,
        version=version,
        unit_length=unit_length,
        text_encode=text_encode,
        text_sum_way=text_sum_way,
        debug=debug,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return loader
