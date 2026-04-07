import os
import torch
from torch.utils import data
import numpy as np

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')


class VQMotionDataset(data.Dataset):
    def __init__(self, dataset_name, feat_bias=5, window_size=64, unit_length=8, motion_type=None, text_type=None, version=None):
        self.mean = np.load(f'{_DATA_DIR}/mean_correct_duet_scalar_rot.npy')
        self.std = np.load(f'{_DATA_DIR}/std_correct_duet_scalar_rot.npy') + 1e-6
        self.std_tensor = torch.from_numpy(self.std)
        self.mean_tensor = torch.from_numpy(self.mean)
        self.data_train = dict(np.load(f'{_DATA_DIR}/train_full_correct_duet_scalar_rot.npz', allow_pickle=True))
        self.data_test = dict(np.load(f'{_DATA_DIR}/test_full_correct_duet_scalar_rot.npz', allow_pickle=True))
        self.id_list_train = list(self.data_train.keys())
        self.id_list_test = list(self.data_test.keys())
        self.id_list = self.id_list_test + self.id_list_train
        self.user = [0] * len(self.id_list_test) + [1] * len(self.id_list_train)
        self.window_size = window_size
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
        use = self.user[item]
        if use == 0:
            motion = self.data_test[name].item()['motion'][:]
        else:
            motion = self.data_train[name].item()['motion'][:]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion, name


def DATALoader(dataset_name,
               batch_size=1,
               num_workers=8, unit_length=4, motion_type=None, text_type=None, version=None):

    dataset = VQMotionDataset(dataset_name, unit_length=unit_length, motion_type=motion_type, text_type=text_type, version=version)
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               pin_memory=True,
                                               drop_last=True)

    return train_loader, dataset.mean, dataset.std
