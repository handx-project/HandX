import os
import torch
from torch.utils import data
import numpy as np
from torch.utils.data._utils.collate import default_collate

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')


def collate_fn(batch):
    return default_collate(batch)


class MotionMillionFSQDataset(data.Dataset):
    def __init__(self, motion_dim, is_test, w_vectorizer, feat_bias=5, max_text_len=20, unit_length=4, version="version1/tokenizer_no_mirror"):
        if motion_dim in [168, 288]:
            self.mean = np.load(f'{_DATA_DIR}/mean_correct_duet_scalar_rot.npy')
            self.std = np.load(f'{_DATA_DIR}/std_correct_duet_scalar_rot.npy') + 1e-6
            self.mean = self.mean[:motion_dim]
            self.std = self.std[:motion_dim]
            self.std_tensor = torch.from_numpy(self.std)
            self.mean_tensor = torch.from_numpy(self.mean)
            self.data = dict(np.load(f'{_DATA_DIR}/test_full_correct_duet_scalar_rot.npz', allow_pickle=True))
        else:
            raise ValueError(f'Unsupported motion_dim: {motion_dim}')

        self.motion_dim = motion_dim
        self.id_list = list(self.data.keys())
        L = len(self.id_list) // 10 + 1
        self.id_list = self.id_list[:L]
        self.unit_length = unit_length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def transform(self, data):
        return (data - self.mean) / self.std

    def inv_transform_torch(self, data):
        device = data.device
        return data * self.std_tensor.to(device) + self.mean_tensor.to(device)

    def transform_torch(self, data):
        device = data.device
        return (data - self.mean_tensor.to(device)) / self.std_tensor.to(device)

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, item):
        name = self.id_list[item]
        motion = self.data[name].item()['motion'][:, :self.motion_dim]
        m_length = len(motion)

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion, m_length, name


def MotionMillionFSQDATALoader(motion_dim, is_test,
                               batch_size, w_vectorizer,
                               num_workers=8, unit_length=4, version="version1/tokenizer_no_mirror"):

    val_dataset = MotionMillionFSQDataset(motion_dim, is_test, w_vectorizer, unit_length=unit_length, version=version)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size,
                                             shuffle=True,
                                             num_workers=40,
                                             collate_fn=collate_fn,
                                             drop_last=True,
                                             prefetch_factor=2)
    return val_loader, val_dataset.mean, val_dataset.std
