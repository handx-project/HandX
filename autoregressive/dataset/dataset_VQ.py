import os
import torch
from torch.utils import data
import numpy as np
import random

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')


class VQMotionDatasetEval(data.Dataset):
    def __init__(self, dataset_name, motion_dim, text_type, version, split, debug, window_size=64, unit_length=4):
        if motion_dim in [168, 288]:
            self.mean = np.load(f'{_DATA_DIR}/mean_correct_duet_scalar_rot.npy')
            self.std = np.load(f'{_DATA_DIR}/std_correct_duet_scalar_rot.npy') + 1e-6
            self.mean = self.mean[:motion_dim]
            self.std = self.std[:motion_dim]
            self.std_tensor = torch.from_numpy(self.std)
            self.mean_tensor = torch.from_numpy(self.mean)
            self.data = dict(np.load(f'{_DATA_DIR}/{split}_full_correct_duet_scalar_rot.npz', allow_pickle=True))
        else:
            raise ValueError(f'Unsupported motion_dim: {motion_dim}')

        self.id_list = list(self.data.keys())
        if debug:
            self.id_list = self.id_list[:]
        self.window_size = window_size
        self.unit_length = unit_length
        self.motion_dim = motion_dim

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
        motion = self.data[name].item()['motion'][..., :self.motion_dim]
        m_length = len(motion)

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion, m_length, name


class VQMotionDataset(data.Dataset):
    def __init__(self, dataset_name, motion_dim, text_type, version, split, debug, window_size=64, unit_length=4):
        if motion_dim in [168, 288]:
            self.mean = np.load(f'{_DATA_DIR}/mean_correct_duet_scalar_rot.npy')
            self.std = np.load(f'{_DATA_DIR}/std_correct_duet_scalar_rot.npy') + 1e-6
            self.mean = self.mean[:motion_dim]
            self.std = self.std[:motion_dim]
            self.std_tensor = torch.from_numpy(self.std)
            self.mean_tensor = torch.from_numpy(self.mean)
            self.data = dict(np.load(f'{_DATA_DIR}/{split}_full_correct_duet_scalar_rot.npz', allow_pickle=True))
        else:
            raise ValueError(f'Unsupported motion_dim: {motion_dim}')

        self.id_list = list(self.data.keys())
        if debug:
            self.id_list = self.id_list[:1500]
        self.window_size = window_size
        self.unit_length = unit_length
        self.motion_dim = motion_dim
        self.indices = self._create_indices()

    def _create_indices(self):
        indices = []
        for i, _ in enumerate(self.id_list):
            for start_idx in range(0, 60 - self.window_size + 1, 1):
                indices.append((i, start_idx))
        return indices

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
        return len(self.indices)

    def __getitem__(self, item):
        sample_idx, time_idx = self.indices[item]
        name = self.id_list[sample_idx]
        motion = self.data[name].item()['motion'][time_idx:time_idx + self.window_size, :self.motion_dim]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion


def DATALoader(dataset_name,
               batch_size,
               motion_type,
               text_type,
               version,
               split,
               debug,
               num_workers=64,
               window_size=64,
               unit_length=4):
    print("num_workers: ", num_workers)
    trainSet = VQMotionDataset(dataset_name, 288, text_type, version, split, debug, window_size=window_size, unit_length=unit_length)
    train_loader = torch.utils.data.DataLoader(trainSet,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               drop_last=True,
                                               pin_memory=True)

    return train_loader, trainSet.mean, trainSet.std


def DATALoaderEvalVQ(dataset_name,
                     batch_size,
                     motion_dim,
                     text_type,
                     version,
                     split,
                     debug,
                     num_workers=64,
                     window_size=64,
                     unit_length=4):
    print("num_workers: ", num_workers)
    trainSet = VQMotionDatasetEval(dataset_name, motion_dim, text_type, version, split, debug, window_size=window_size, unit_length=unit_length)
    train_loader = torch.utils.data.DataLoader(trainSet,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=num_workers,
                                               drop_last=False,
                                               pin_memory=True)

    return train_loader, trainSet.mean, trainSet.std
