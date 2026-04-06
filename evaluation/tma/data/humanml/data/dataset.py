import codecs as cs
import os
import random
from os.path import join as pjoin

import numpy as np
import spacy
import torch
from rich.progress import track
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from ..utils.get_opt import get_opt
from ..utils.word_vectorizer import WordVectorizer
import json

import io, json, lmdb, numpy as np, torch
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
def _is_npy(buf: bytes) -> bool:
    return len(buf) >= 6 and buf[:6] == b"\x93NUMPY"

def _load_value(buf: bytes):
 
    if _is_npy(buf):
        return np.load(io.BytesIO(buf), allow_pickle=False)
    return json.loads(buf.decode("utf-8"))
def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)




class Text2MotionDataset(data.Dataset):

    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        min_motion_len = 40 if self.opt.dataset_name == "t2m" else 24

        joints_num = opt.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + ".npy"))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20):int(to_tag *
                                                                      20)]
                                if (len(n_motion)) < min_motion_len or (
                                        len(n_motion) >= 200):
                                    continue
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
                                    "_" + name)
                                while new_name in data_dict:
                                    new_name = (random.choice(
                                        "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
                                                name)
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                      to_tag, name)

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                # Some motion may not exist in KIT dataset
                pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if opt.is_train:
            std[0:1] = std[0:1] / opt.feat_bias
            std[1:3] = std[1:3] / opt.feat_bias
            std[3:4] = std[3:4] / opt.feat_bias
            std[4:4 + (joints_num - 1) * 3] = std[4:4 +
                                                  (joints_num - 1) * 3] / 1.0
            std[4 + (joints_num - 1) * 3:4 +
                (joints_num - 1) * 9] = (std[4 + (joints_num - 1) * 3:4 +
                                             (joints_num - 1) * 9] / 1.0)
            std[4 + (joints_num - 1) * 9:4 + (joints_num - 1) * 9 +
                joints_num * 3] = (std[4 + (joints_num - 1) * 9:4 +
                                       (joints_num - 1) * 9 + joints_num * 3] /
                                   1.0)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = (
                std[4 +
                    (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias)

            assert 4 + (joints_num -
                        1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, "mean.npy"), mean)
            np.save(pjoin(opt.meta_dir, "std.npy"), std)

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.opt.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data[
            "text"]
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.opt.max_text_len:
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.opt.max_text_len + 2 - sent_len)
        else:
            tokens = tokens[:self.opt.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if self.opt.is_train:
            if m_length != self.max_length:
                if self.opt.unit_length < 10:
                    coin2 = np.random.choice(["single", "single", "double"])
                else:
                    coin2 = "single"
                if len_gap == 0 or (len_gap == 1 and coin2 == "double"):
                    m_length = self.max_length
                    idx = random.randint(0, m_length - self.max_length)
                    motion = motion[idx:idx + self.max_length]
                else:
                    if coin2 == "single":
                        n_m_length = self.max_length + self.opt.unit_length * len_gap
                    else:
                        n_m_length = self.max_length + self.opt.unit_length * (
                            len_gap - 1)
                    idx = random.randint(0, m_length - n_m_length)
                    motion = motion[idx:idx + self.max_length]
                    m_length = n_m_length
                # print(len_gap, idx, coin2)
        else:
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(["single", "single", "double"])
            else:
                coin2 = "single"

            if coin2 == "double":
                m_length = (m_length // self.opt.unit_length -
                            1) * self.opt.unit_length
            elif coin2 == "single":
                m_length = (m_length //
                            self.opt.unit_length) * self.opt.unit_length
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx + m_length]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length



"For Dataset UniMocap"

class UniMocapDataset(data.Dataset):

    def __init__(
        self,
        mean,
        std,
        split_file,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        motion_dir,
        text_dir,
        input_format, 
        njoints, 
        text_source, # new args
        tiny=False,
        debug=False,
        progress_bar=True,
        **kwargs,
    ):
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length

        self.text_source = text_source
        
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        self.id_list = id_list

        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10

        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading {split_file.split('/')[-2]} {split_file.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)
        count = 0
        bad_count = 0
        miss_count = 0
        new_name_list = []
        length_list = []
        for i, name in enumerator:
            if count > maxdata:
                break
            try:
                motion = np.load(pjoin(motion_dir, name + ".npy"))
                if np.any(np.isnan(motion)):
                    bad_count += 1
                    continue
                if input_format == 'root_position':
                    motion = motion[..., :4+(njoints-1)*3]
                elif input_format == 'root_position_vel':
                    motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=-1)
                elif input_format == 'root_position_rot6d':
                    motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 3: 4+(njoints - 1) * 9]), axis=-1)
                elif input_format == 'root_rot6d':
                    motion = np.concatenate((motion[..., :4], motion[..., 4+(njoints - 1) * 3: 4+(njoints - 1) * 9]), axis=-1)
                elif input_format == 'vector_263':
                    pass
                else:
                    print('NotImplementedError')
                    raise NotImplementedError

                if (len(motion)) < self.min_motion_length or (len(motion) >=
                                                                200):
                    bad_count += 1
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        if "#" not in line:
                            f_tag = 0.0
                            to_tag = 0.0
                            caption = line
                            tokens = line.split(" ")
                        else:
                            line_split = line.strip().split("#")
                            caption = line_split[0]
                            try:
                                tokens = line_split[1].split(" ")
                            except:
                                import pdb; pdb.set_trace()
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20):int(to_tag *
                                                                        20)]
                                if (len(n_motion)
                                    ) < self.min_motion_length or (
                                        (len(n_motion) >= 200)):
                                    
                                    continue
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
                                    "_" + name)
                                while new_name in data_dict:
                                    new_name = (random.choice(
                                        "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
                                                name)
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                        to_tag, name)

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
                    count += 1
            except:
                miss_count += 1
                pass

        print(f'Here are {miss_count} not in dataset!')
        print(f'Here are {bad_count} either small or large than given value.')

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))



        
        self.mean = mean
        self.std = std

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = motion.shape[1]
        self.name_list = name_list
        self.reset_max_len(self.max_length)

        
        print('train len', len(data_dict))
        print('test len', len(data_dict))
    

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]

        retrieval_name = self.name_list[idx].split('_')[-1]

        motion, m_length, text_list = data["motion"], data["length"], data[
            "text"]

        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if self.text_source == 'token':
            if len(tokens) < self.max_text_len:
                tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
                sent_len = len(tokens)
                tokens = tokens + ["unk/OTHER"] * (self.max_text_len + 2 - sent_len)
            else:
                tokens = tokens[:self.max_text_len]
                tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
                sent_len = len(tokens)
        elif self.text_source == 'only_text_token' or self.text_source == 'caption':

            if len(tokens) < self.max_text_len:
                tokens = ['sos'] + tokens + ['eos']
                sent_len = len(tokens)
                tokens = tokens + ['unk'] * (self.max_text_len + 2 - sent_len)
            else:
                tokens = tokens[:self.max_text_len]
                tokens = ['sos'] + tokens + ['eos']
                sent_len = len(tokens)

        if self.text_source == 'token':
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(pos_oh[None, :])
                word_embeddings.append(word_emb[None, :])
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)
        elif self.text_source == 'only_text_token':
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb = self.w_vectorizer[token]
                word_embeddings.append(word_emb[None, :])
            word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        "Z Normalization"
        motion = (motion - self.mean) / (self.std + 1e-7)

        if np.any(np.isnan(motion)):
            print(retrieval_name, "nan in motion")
            motion = np.random.rand(*(motion.shape))
            raise ValueError("nan in motion")

        return (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            "_".join(tokens),
            retrieval_name
        )



class Text2MotionDatasetBaseline(data.Dataset):

    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name == "t2m" else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + ".npy"))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20):int(to_tag *
                                                                      20)]
                                if (len(n_motion)) < min_motion_len or (
                                        len(n_motion) >= 200):
                                    continue
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
                                    "_" + name)
                                while new_name in data_dict:
                                    new_name = (random.choice(
                                        "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
                                                name)
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                      to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = motion.shape[1]
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data[
            "text"]
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.opt.max_text_len:
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.opt.max_text_len + 2 - sent_len)
        else:
            tokens = tokens[:self.opt.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if m_length != self.max_length:
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(["single", "single", "double"])
            else:
                coin2 = "single"
            if len_gap == 0 or (len_gap == 1 and coin2 == "double"):
                m_length = self.max_length
                s_idx = random.randint(0, m_length - self.max_length)
            else:
                if coin2 == "single":
                    n_m_length = self.max_length + self.opt.unit_length * len_gap
                else:
                    n_m_length = self.max_length + self.opt.unit_length * (
                        len_gap - 1)
                s_idx = random.randint(0, m_length - n_m_length)
                m_length = n_m_length
        else:
            s_idx = 0

        src_motion = motion[s_idx:s_idx + m_length]
        tgt_motion = motion[s_idx:s_idx + self.max_length]
        "Z Normalization"
        src_motion = (src_motion - self.mean) / self.std
        tgt_motion = (tgt_motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            src_motion = np.concatenate(
                [
                    src_motion,
                    np.zeros(
                        (self.max_motion_length - m_length, motion.shape[1])),
                ],
                axis=0,
            )
        
        return word_embeddings, caption, sent_len, src_motion, tgt_motion, m_length


class MotionDatasetV2(data.Dataset):

    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        joints_num = opt.joints_num

        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + ".npy"))
                if motion.shape[0] < opt.window_size:
                    continue
                self.lengths.append(motion.shape[0] - opt.window_size)
                self.data.append(motion)
            except:
                pass

        self.cumsum = np.cumsum([0] + self.lengths)

        if opt.is_train:
            std[0:1] = std[0:1] / opt.feat_bias
            std[1:3] = std[1:3] / opt.feat_bias
            std[3:4] = std[3:4] / opt.feat_bias
            std[4:4 + (joints_num - 1) * 3] = std[4:4 +
                                                  (joints_num - 1) * 3] / 1.0
            std[4 + (joints_num - 1) * 3:4 +
                (joints_num - 1) * 9] = (std[4 + (joints_num - 1) * 3:4 +
                                             (joints_num - 1) * 9] / 1.0)
            std[4 + (joints_num - 1) * 9:4 + (joints_num - 1) * 9 +
                joints_num * 3] = (std[4 + (joints_num - 1) * 9:4 +
                                       (joints_num - 1) * 9 + joints_num * 3] /
                                   1.0)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = (
                std[4 +
                    (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias)

            assert 4 + (joints_num -
                        1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, "mean.npy"), mean)
            np.save(pjoin(opt.meta_dir, "std.npy"), std)

        self.mean = mean
        self.std = std
        print("Total number of motions {}, snippets {}".format(
            len(self.data), self.cumsum[-1]))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx + self.opt.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion


class RawTextDataset(data.Dataset):

    def __init__(self, opt, mean, std, text_file, w_vectorizer):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.nlp = spacy.load("en_core_web_sm")

        with cs.open(text_file) as f:
            for line in f.readlines():
                word_list, pos_list = self.process_text(line.strip())
                tokens = [
                    "%s/%s" % (word_list[i], pos_list[i])
                    for i in range(len(word_list))
                ]
                self.data_dict.append({
                    "caption": line.strip(),
                    "tokens": tokens
                })

        self.w_vectorizer = w_vectorizer
        print("Total number of descriptions {}".format(len(self.data_dict)))

    def process_text(self, sentence):
        sentence = sentence.replace("-", "")
        doc = self.nlp(sentence)
        word_list = []
        pos_list = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == "NOUN"
                    or token.pos_ == "VERB") and (word != "left"):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[item]
        caption, tokens = data["caption"], data["tokens"]

        if len(tokens) < self.opt.max_text_len:
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.opt.max_text_len + 2 - sent_len)
        else:
            tokens = tokens[:self.opt.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len


class TextOnlyDataset(data.Dataset):

    def __init__(self, opt, mean, std, split_file, text_dir, **kwargs):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.max_length = 20
        self.pointer = 0
        self.fixed_length = 120

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                text_data = []
                flag = False
                with cs.open(pjoin(text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
                                    "_" + name)
                                while new_name in data_dict:
                                    new_name = (random.choice(
                                        "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
                                                name)
                                data_dict[new_name] = {"text": [text_dict]}
                                new_name_list.append(new_name)
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                      to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {"text": text_data}
                    new_name_list.append(name)
            except:
                pass

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        text_list = data["text"]

        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]
        return None, None, caption, None, np.array([0
                                                    ]), self.fixed_length, None


class HumanML3D(data.Dataset):

    def __init__(self,
                 mode,
                 datapath="./dataset/humanml_opt.txt",
                 split="train",
                 **kwargs):
        self.mode = mode

        self.dataset_name = "t2m"
        self.dataname = "t2m"

        abs_base_path = f"."
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = (
            None 
        )
        opt = get_opt(dataset_opt_path, device)
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        self.opt = opt
        print("Loading dataset %s ..." % opt.dataset_name)

        if mode == "gt":
            self.mean = np.load(pjoin(opt.meta_dir, "mean.npy"))
            self.std = np.load(pjoin(opt.meta_dir, "std.npy"))
        elif mode in ["train", "eval", "text_only"]:
            self.mean = np.load(pjoin(opt.data_root, "Mean.npy"))
            self.std = np.load(pjoin(opt.data_root, "Std.npy"))

        if mode == "eval":
          
            self.mean_for_eval = np.load(pjoin(opt.meta_dir, "mean.npy"))
            self.std_for_eval = np.load(pjoin(opt.meta_dir, "std.npy"))

        self.split_file = pjoin(opt.data_root, f"{split}.txt")
        if mode == "text_only":
            self.t2m_dataset = TextOnlyDataset(self.opt, self.mean, self.std,
                                               self.split_file)
        else:
            self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, "glove"),
                                               "our_vab")
            self.t2m_dataset = Text2MotionDatasetV2(self.opt, self.mean,
                                                    self.std, self.split_file,
                                                    self.w_vectorizer)
            self.num_actions = 1  

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()


class KIT(HumanML3D):

    def __init__(self,
                 mode,
                 datapath="./dataset/kit_opt.txt",
                 split="train",
                 **kwargs):
        super(KIT, self).__init__(mode, datapath, split, **kwargs)


'''For Motion-X'''
class Text2MotionDatasetMotionX(data.Dataset):
    def __init__(
        self,
        mean,
        std,
        split_file,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        motion_dir,
        semantic_text_dir,
        face_text_dir,
        condition,
        dataset_name,
        eval_text_encode_way, 
        text_source, 
        motion_type, 
        tiny=False,
        debug=False,
        progress_bar=True,
        **kwargs):
        
        
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        self.max_text_len = max_text_len
        self.dataset_name = dataset_name
        self.text_source = text_source
        self.eval_text_encode_way = eval_text_encode_way
        self.unit_length = unit_length

        if eval_text_encode_way == 'clip':
            text_enc, clip_preprocess = clip.load("ViT-B/32", device=opt.device, jit=False)  # Must set jit=False for training
            clip.model.convert_weights(text_enc)# Actually this line is unnecessary since clip by default already on float16
            self.tokenizer = clip.tokenize
            text_enc.eval()
            for p in text_enc.parameters():
                p.requires_grad = False
            self.text_enc = text_enc

        elif eval_text_encode_way == 't5':
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            text_enc = SentenceTransformer('sentence-transformers/sentence-t5-xl').to(opt.device)
            text_enc.eval()
            for p in text_enc.parameters():
                p.requires_grad = False
            self.text_enc = text_enc


        if dataset_name =='t2m' or dataset_name =='motionx':
            min_motion_len = 40 
        else:
            min_motion_len = 24


        data_dict = {}
        id_list = []

        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10
        
        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading {self.dataset_name} {split_file.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)

        
        count = 0
        new_name_list = []
        length_list = []
        for i, name in enumerator:
            if count > maxdata:
                break
            
            # try:
            motion = np.load(pjoin(motion_dir, motion_type, name + '.npy'))
            if (len(motion)) < min_motion_len:
                continue
            elif len(motion) >= self.max_motion_length:
                start = random.randint(0,len(motion) - self.max_motion_length)
                motion = motion[start:start+self.max_motion_length]
            text_data = []
            flag = False
            with cs.open(pjoin(semantic_text_dir, name + '.txt')) as f:
                for line in f.readlines():
                    if 'humanml' in name:
                        if text_source == 'token':
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                try:
                                    n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                    if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                        continue
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    while new_name in data_dict:
                                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    data_dict[new_name] = {'motion': n_motion,
                                                        'length': len(n_motion),
                                                        'text':[text_dict]}
                                    new_name_list.append(new_name)
                                    length_list.append(len(n_motion))
                                except:
                                    print(line_split)
                                    print(line_split[2], line_split[3], f_tag, to_tag, name)
                                    # break
                        elif text_source in ['only_text_token',  'caption']:
                            
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            tokens = [i.split('/')[0] for i in line_split[1].split(' ')]
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                try:
                                    n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                    if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                        continue
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    while new_name in data_dict:
                                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    data_dict[new_name] = {'motion': n_motion,
                                                        'length': len(n_motion),
                                                        'text':[text_dict]}
                                    new_name_list.append(new_name)
                                    length_list.append(len(n_motion))
                                except:
                                    print(line_split)
                                    print(line_split[2], line_split[3], f_tag, to_tag, name)
                                    # break
                        else:
                            raise NotImplementedError


                    else:
                        text_dict = {}
                        line_split = line.strip()
                        caption = line_split
                        tokens = caption.split(' ')
                        f_tag = 0.0 
                        to_tag = 0.0

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)



            if flag:
                data_dict[name] = {'motion': motion,
                                    'length': len(motion),
                                    'text': text_data}
                new_name_list.append(name)
                length_list.append(len(motion))
                count += 1
            # except:
            #     import pdb; pdb.set_trace()
            #     pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        self.nfeats = motion.shape[1]
        

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * (self.std + 1e-7) + self.mean

    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        
        retrieval_name = self.name_list[idx]
        
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']


        if self.text_source == 'token':
            if len(tokens) < self.max_text_len:
                # pad with "unk"
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
        elif self.text_source == 'only_text_token' or self.text_source == 'caption':

            if len(tokens) < self.max_text_len:
                # pad with "unk"
                tokens = ['sos'] + tokens + ['eos']
                sent_len = len(tokens)
                tokens = tokens + ['unk'] * (self.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.max_text_len]
                tokens = ['sos'] + tokens + ['eos']
                sent_len = len(tokens)


        if self.text_source == 'token':
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(pos_oh[None, :])
                word_embeddings.append(word_emb[None, :])
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)
        elif self.text_source == 'only_text_token':
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb = self.w_vectorizer[token]
                word_embeddings.append(word_emb[None, :])
            word_embeddings = np.concatenate(word_embeddings, axis=0)

        elif self.text_source == 'caption':
            pos_one_hots = []
            word_embeddings = []

            for token in tokens:
                if self.eval_text_encode_way == 'clip':
                    token = self.tokenizer(token, truncate=True).to(self.opt.device) 
                    word_emb = self.text_enc.encode_text(token).squeeze().cpu().detach().numpy() # (512,)
                elif self.eval_text_encode_way == 't5':
                    word_emb = self.text_enc.encode(token) # 
                else:
                    word_emb = self.w_vectorizer[token] # (300,)
                    

                word_embeddings.append(word_emb[None, :])

            word_embeddings = np.concatenate(word_embeddings, axis=0)


        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / (self.std + 1e-7)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), retrieval_name




'''For Motion-X'''
class Text2MotionDatasetMotionX_text_all(data.Dataset):
    def __init__(
        self,
        mean,
        std,
        split_file,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        motion_dir,
        semantic_text_dir,
        face_text_dir,
        dataset_name,
        eval_text_encode_way, 
        text_source, 
        motion_type, 
        condition, 
        tiny=False,
        debug=False,
        progress_bar=True,
        **kwargs):
        
        
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        self.max_text_len = max_text_len
        self.dataset_name = dataset_name
        self.text_source = text_source
        self.eval_text_encode_way = eval_text_encode_way
        self.unit_length = unit_length
        self.condition = condition
        assert self.condition in ['text_all', 'text_body', 'text_hand', 'text_face', 'text_face_body', 'text_seperate', 'only_pose_concat', 'only_pose_fusion']

        if eval_text_encode_way == 'clip':
            text_enc, clip_preprocess = clip.load("ViT-B/32", device=opt.device, jit=False)  # Must set jit=False for training
            clip.model.convert_weights(text_enc)# Actually this line is unnecessary since clip by default already on float16
            self.tokenizer = clip.tokenize
            text_enc.eval()
            for p in text_enc.parameters():
                p.requires_grad = False
            self.text_enc = text_enc

        elif eval_text_encode_way == 't5':
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            text_enc = SentenceTransformer('sentence-transformers/sentence-t5-xl').to(opt.device)
            text_enc.eval()
            for p in text_enc.parameters():
                p.requires_grad = False
            self.text_enc = text_enc


        if dataset_name =='t2m' or dataset_name =='motionx':
            min_motion_len = 40 
        else:
            min_motion_len = 24


        data_dict = {}
        id_list = []

        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        
        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10
        
        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading {self.dataset_name} {split_file.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)

        
        count = 0
        new_name_list = []
        length_list = []
        for i, name in enumerator:
            if count > maxdata:
                break
            
            try:
                motion = np.load(pjoin(motion_dir, motion_type, name + '.npy'))

                if (len(motion)) < min_motion_len:
                    continue
                elif len(motion) >= self.max_motion_length:
                    start = random.randint(0,len(motion) - self.max_motion_length)
                    motion = motion[start:start+self.max_motion_length]
                text_data = []
                flag = False
                with cs.open(pjoin(semantic_text_dir, name + '.txt')) as f:
                    
                    try:
                        face_f = open(pjoin(face_text_dir, name + '.txt'))
                        face_text = face_f.readlines()[0]
                    except:
                        import pdb; pdb.set_trace()

                    with open(pjoin(face_text_dir.replace('face_texts', 'body_texts'), name + '.json'), 'r') as body_f:
                        body_dict = json.load(body_f)

                    with open(pjoin(face_text_dir.replace('face_texts', 'hand_texts'), name + '.json'), 'r') as hand_f:
                        hand_dict = json.load(hand_f)


                    for line in f.readlines():
                        if 'humanml' in name:
                            if text_source == 'token':
                                text_dict = {}
                                line_split = line.strip().split('#')
                                caption = line_split[0]
                                tokens = line_split[1].split(' ')
                                f_tag = float(line_split[2])
                                to_tag = float(line_split[3])
                                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                                text_dict['caption'] = caption
                                text_dict['tokens'] = tokens
                                text_dict['face'] = face_text
                                text_dict['hand'] = hand_dict
                                text_dict['body'] = body_dict

                                if f_tag == 0.0 and to_tag == 0.0:
                                    flag = True
                                    text_data.append(text_dict)
                                else:
                                    try:
                                        n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                        if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                            continue
                                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                        while new_name in data_dict:
                                            new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                        data_dict[new_name] = {'motion': n_motion,
                                                            'length': len(n_motion),
                                                            'text':[text_dict]}
                                        new_name_list.append(new_name)
                                        length_list.append(len(n_motion))
                                    except:
                                        print(line_split)
                                        print(line_split[2], line_split[3], f_tag, to_tag, name)
                                        # break
                            elif text_source in ['only_text_token',  'caption']:
                                
                                text_dict = {}
                                line_split = line.strip().split('#')
                                caption = line_split[0]
                                tokens = [i.split('/')[0] for i in line_split[1].split(' ')]
                                f_tag = float(line_split[2])
                                to_tag = float(line_split[3])
                                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                                text_dict['caption'] = caption
                                text_dict['tokens'] = tokens
                                text_dict['face'] = face_text
                                text_dict['hand'] = hand_dict
                                text_dict['body'] = body_dict

                                if f_tag == 0.0 and to_tag == 0.0:
                                    flag = True
                                    text_data.append(text_dict)
                                else:
                                    try:
                                        n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                        if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                            continue
                                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                        while new_name in data_dict:
                                            new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                        data_dict[new_name] = {'motion': n_motion,
                                                            'length': len(n_motion),
                                                            'text':[text_dict]}
                                        new_name_list.append(new_name)
                                        length_list.append(len(n_motion))
                                    except:
                                        print(line_split)
                                        print(line_split[2], line_split[3], f_tag, to_tag, name)
                                        # break
                            else:
                                raise NotImplementedError
   

                        else:
                            text_dict = {}
                            line_split = line.strip()
                            caption = line_split
                            tokens = caption.split(' ')
                            f_tag = 0.0 
                            to_tag = 0.0

                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            text_dict['face'] = face_text
                            text_dict['hand'] = hand_dict
                            text_dict['body'] = body_dict

                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)



                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
                    count += 1
            except:
                import pdb; pdb.set_trace()
                pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        self.nfeats = motion.shape[1]
        

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * (self.std + 1e-7) + self.mean

    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        face_text, hand_dict, body_dict = text_data['face'], text_data["hand"],  text_data["body"]

        select_frame = random.randint(0, len(hand_dict)-1)
        hand_frame_data, body_frame_data = hand_dict[str(select_frame)], body_dict[str(select_frame)]

        body_text = random.choice(body_frame_data.split('.')[:-1]) + '.'
        hand_text = random.choice(hand_frame_data.split('.')[:-1]) + '.'

        if self.text_source == 'token':
            if len(tokens) < self.max_text_len:
                # pad with "unk"
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
        elif self.text_source == 'only_text_token' or self.text_source == 'caption':

            if len(tokens) < self.max_text_len:
                # pad with "unk"
                tokens = ['sos'] + tokens + ['eos']
                sent_len = len(tokens)
                tokens = tokens + ['unk'] * (self.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.max_text_len]
                tokens = ['sos'] + tokens + ['eos']
                sent_len = len(tokens)


        if self.text_source == 'token':
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(pos_oh[None, :])
                word_embeddings.append(word_emb[None, :])
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)
        elif self.text_source == 'only_text_token':
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb = self.w_vectorizer[token]
                word_embeddings.append(word_emb[None, :])
            word_embeddings = np.concatenate(word_embeddings, axis=0)

        elif self.text_source == 'caption':
            pos_one_hots = []
            word_embeddings = []

            for token in tokens:
                if self.eval_text_encode_way == 'clip':
                    token = self.tokenizer(token, truncate=True).to(self.opt.device) 
                    word_emb = self.text_enc.encode_text(token).squeeze().cpu().detach().numpy() # (512,)
                elif self.eval_text_encode_way == 't5':
                    word_emb = self.text_enc.encode(token) # 
                else:
                    word_emb = self.w_vectorizer[token] # (300,)
                    

                word_embeddings.append(word_emb[None, :])

            word_embeddings = np.concatenate(word_embeddings, axis=0)


        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / (self.std + 1e-7)
        
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), body_text, hand_text, face_text
    
    
    
"""For use of training text motion matching model, and evaluations"""


class Text2MotionDatasetV2(data.Dataset):
    def __init__(
        self,
        mean,
        std,
        split_file,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        motion_dir,
        text_dir,
        input_format, 
        njoints, 
        tiny=False,
        debug=False,
        progress_bar=True,
        **kwargs,
    ):
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        # min_motion_len = 40 if dataset_name =='t2m' else 24

        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        self.id_list = id_list

        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10

        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading {split_file.split('/')[-2]} {split_file.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)
        count = 0
        bad_count = 0
        miss_count = 0
        new_name_list = []
        length_list = []
        print(motion_dir,'MOTION_DIR')
        
        
        
        for i, name in enumerator:
            if count > maxdata:
                break
            try:
                motion = np.load(pjoin(motion_dir, name + ".npy"))

                if input_format == 'root_position':
                    motion = motion[..., :4+(njoints-1)*3]
                elif input_format == 'root_position_vel':
                    motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 9: 4+(njoints - 1) * 9 + njoints*3]), axis=-1)
                elif input_format == 'root_position_rot6d':
                    motion = np.concatenate((motion[..., :4+(njoints - 1) * 3], motion[..., 4+(njoints - 1) * 3: 4+(njoints - 1) * 9]), axis=-1)
                elif input_format == 'root_rot6d':
                    motion = np.concatenate((motion[..., :4], motion[..., 4+(njoints - 1) * 3: 4+(njoints - 1) * 9]), axis=-1)
                elif input_format == 'vector_263':
                    pass
                else:
                    print('NotImplementedError')
                    raise NotImplementedError

                if (len(motion)) < self.min_motion_length or (len(motion) >=
                                                                200):
                    bad_count += 1
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        try:
                            tokens = line_split[1].split(" ")
                        except:
                            import pdb; pdb.set_trace()
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20):int(to_tag *
                                                                        20)]
                                if (len(n_motion)
                                    ) < self.min_motion_length or (
                                        (len(n_motion) >= 200)):
                                    
                                    continue
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
                                    "_" + name)
                                while new_name in data_dict:
                                    new_name = (random.choice(
                                        "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
                                                name)
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                # None
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                        to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
                    # print(count)
                    count += 1
                    # print(name)
            except:
                # import pdb; pdb.set_trace()
                miss_count += 1
                pass

        print(f'Here are {miss_count} not in dataset!')
        print(f'Here are {bad_count} either small or large than given value.')

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))



        
        self.mean = mean
        self.std = std

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = motion.shape[1]
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        # train 24546
        # test 4648
        print('train len', len(data_dict))
        print('test len', len(data_dict))



    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]

        retrieval_name = self.name_list[idx].split('_')[-1]

        motion, m_length, text_list = data["motion"], data["length"], data[
            "text"]

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        # debug check nan
        if np.any(np.isnan(motion)):
            raise ValueError("nan in motion")

        return (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            "_".join(tokens),
            retrieval_name
        )
        # return caption, motion, m_length

class Text2MotionDatasetV3_old_wolmdb(data.Dataset):
    def __init__(
        self,
        mean,
        std,
        split_datapart,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        motion_dir,
        text_dir,
        input_format, 
        njoints, 
        tiny=False,
        debug=False,
        progress_bar=True,
        **kwargs,
    ):
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        # min_motion_len = 40 if dataset_name =='t2m' else 24

        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length

        data_dict = {}
        id_list = []
        split_root='/scratch/bcnt/sirui/OpenTMA/dataset_split2'
        
        for DataSet in ['behave', 'intercap', 'neuraldome', 'grab', 'chairs', 'omomo', 'imhd']:
            txt_name=os.path.join(split_root,DataSet+'_'+split_datapart+'.txt')
            with open(txt_name,'r') as f:
                names_text_list=f.readlines()
            id_list_part=[['/scratch/bcnt/sirui/dongting/data',DataSet,e.strip()] for e in names_text_list]
            id_list+=id_list_part
    
        print(len(id_list),'Amount of sequences')

        self.id_list = id_list

        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10

        
        # else:
        enumerator = enumerate(id_list)
            
        count = 0
        bad_count = 0
        miss_count = 0
        new_name_list = []
        length_list = []
        print(motion_dir,'MOTION_DIR')
        
        
        joints_index =list(range(22))+list(range(25,55)) 

        for i, dataset_plus_id in enumerator:
            motion_dir=dataset_plus_id[0]
            
            motion_dir ='/work/hdd/bcnt/sirui/dongting/data'
            DataSet_name=dataset_plus_id[1]
            id_name=dataset_plus_id[2]
            name=DataSet_name+'_'+id_name
            if count > maxdata:
                break
            try:
             
                motion_bps = np.load(pjoin(motion_dir, DataSet_name,'sequences_canonical',id_name,  "bps_time.npy"))

                MP=''
                if DataSet_name in ['omomo','behave','intercap']:
                    DataSet_name_used = DataSet_name+'_correct'
                else:
                    DataSet_name_used = DataSet_name
                    
                DATA_O = dict(np.load(pjoin(MP, DataSet_name_used,'sequences_canonical',id_name,  "data6.npz"),allow_pickle=True))
                
                idxx = list(range(22)) + [24+3*i for i in range(10)]
                
                motion_full =DATA_O['motion']
                motion_h = motion_full[:,:52*3].reshape(-1,52,3)
                
                motion_h = motion_h[:].reshape(motion_h.shape[0],-1)
                
                motion_o = motion_full[:,52*3+52*6+3+4:52*3+52*6+3+9+4].reshape(motion_h.shape[0],-1)
                
                L1 = motion_h.shape[0]
                L2 = motion_bps.shape[0]
                ml= min(L1,L2)
                motion = np.concatenate([motion_h[:ml],motion_o[:ml],motion_bps[:ml]],-1)
                
                motion = motion[:min(300,motion.shape[0]),]
                

                if (len(motion)) < self.min_motion_length or (len(motion) >
                                                                300):
                    bad_count += 1
                    continue
                text_data = []
                flag = False

                with cs.open(pjoin(motion_dir, DataSet_name,'sequences_canonical',id_name,  "text.txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        try:
                            tokens = line_split[1].split(" ")
                        except:
                            import pdb; pdb.set_trace()
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict['sent_len']=len(tokens)
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20):int(to_tag *
                                                                        20)]
                                if (len(n_motion)
                                    ) < self.min_motion_length or (
                                        (len(n_motion) >= 200)):
                                    
                                    continue
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
                                    "_" + name)
                                while new_name in data_dict:
                                    new_name = (random.choice(
                                        "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
                                                name)
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                                #print(new_name,name,'NEW_NAME')
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                # None
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                        to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
                
                    count += 1
            except Exception as e:
                print(e)
                
                miss_count += 1
                pass

        print(f'Here are {miss_count} not in dataset!')
        print(f'Here are {bad_count} either small or large than given value.')

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        print(len(length_list),'LENGTH HERE')



        
        self.mean = mean
        self.std = std

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = motion.shape[1]
        self.name_list = name_list
        self.reset_max_len(self.max_length)

        print(f'Split datapart:{split_datapart}',f'Tiny:{tiny}',len(name_list))
   


    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]

        retrieval_name = self.name_list[idx].split('_')[-1]

        motion, m_length, text_list = data["motion"], data["length"], data[
            "text"]

        text_data = random.choice(text_list)
        caption=text_data["caption"]
        sent_len=text_data['sent_len']
      
   
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if np.any(np.isnan(motion)):
            raise ValueError("nan in motion")

        return (
            caption,
            sent_len,
            motion,
            m_length,
            retrieval_name
        )


class Text2MotionDatasetV3_o(data.Dataset):
    def __init__(
        self,
        mean,
        std,
        split_datapart,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        motion_dir,
        text_dir,
        input_format, 
        njoints, 
        tiny=False,
        debug=False,
        progress_bar=True,
        **kwargs,
    ):
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length

        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length

        data_dict = {}
        id_list = []
        split_root='/scratch/bcnt/sirui/OpenTMA/dataset_split2'
        
        for DataSet in ['behave', 'intercap', 'neuraldome', 'grab', 'chairs', 'omomo', 'imhd']:
            txt_name=os.path.join(split_root,DataSet+'_'+split_datapart+'.txt')
            with open(txt_name,'r') as f:
                names_text_list=f.readlines()
            id_list_part=[['/scratch/bcnt/sirui/dongting/data',DataSet,e.strip()] for e in names_text_list]
            id_list+=id_list_part
      
        print(len(id_list),'Amount of sequences')

        self.id_list = id_list

        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10

       
        enumerator = enumerate(id_list)
            
        count = 0
        bad_count = 0
        miss_count = 0
        new_name_list = []
        length_list = []
        print(motion_dir,'MOTION_DIR')
        
        
        joints_index =list(range(22))+list(range(25,55)) 

        for i, dataset_plus_id in enumerator:
            motion_dir=dataset_plus_id[0]
            
            motion_dir ='/work/hdd/bcnt/sirui/dongting/data'
            DataSet_name=dataset_plus_id[1]
            id_name=dataset_plus_id[2]
            name=DataSet_name+'_'+id_name
            if count > maxdata:
                break
            try:
             
                motion_bps = np.load(pjoin(motion_dir, DataSet_name,'sequences_canonical',id_name,  "bps_time.npy"))

                
                MP=''
                if DataSet_name in ['omomo','behave','intercap']:
                    DataSet_name_used = DataSet_name+'_correct'
                else:
                    DataSet_name_used = DataSet_name
                    
                DATA_O = dict(np.load(pjoin(MP, DataSet_name_used,'sequences_canonical',id_name,  "data6.npz"),allow_pickle=True))
                
                idxx = list(range(22)) + [24+3*i for i in range(10)]
                
                motion_full =DATA_O['motion']
                motion_h = motion_full[:,:52*3].reshape(-1,52,3)
                
                motion_h = motion_h[:].reshape(motion_h.shape[0],-1)
                
                motion_o = motion_full[:,52*3+52*6+3+4:52*3+52*6+3+9+4].reshape(motion_h.shape[0],-1)
                
                L1 = motion_h.shape[0]
                L2 = motion_bps.shape[0]
                ml= min(L1,L2)
                motion = np.concatenate([motion_h[:ml],motion_o[:ml],motion_bps[:ml]],-1)
                
                motion = motion[:min(300,motion.shape[0]),]
                
              

                if (len(motion)) < self.min_motion_length or (len(motion) >
                                                                300):
                    bad_count += 1
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(motion_dir, DataSet_name,'sequences_canonical',id_name,  "text.txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        try:
                            tokens = line_split[1].split(" ")
                        except:
                            import pdb; pdb.set_trace()
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict['sent_len']=len(tokens)
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20):int(to_tag *
                                                                        20)]
                                if (len(n_motion)
                                    ) < self.min_motion_length or (
                                        (len(n_motion) >= 200)):
                                    
                                    continue
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
                                    "_" + name)
                                while new_name in data_dict:
                                    new_name = (random.choice(
                                        "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
                                                name)
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                        to_tag, name)

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
                
                    count += 1
            except Exception as e:
                print(e)
                
                miss_count += 1
                pass

        print(f'Here are {miss_count} not in dataset!')
        print(f'Here are {bad_count} either small or large than given value.')

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        print(len(length_list),'LENGTH HERE')



        
        self.mean = mean
        self.std = std

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = motion.shape[1]
        self.name_list = name_list
        self.reset_max_len(self.max_length)

        print(f'Split datapart:{split_datapart}',f'Tiny:{tiny}',len(name_list))
      



    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.name_list) - self.pointer
    def inv_transform(self, data):
        return data * self.std.reshape(-1) + self.mean.reshape(-1)

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]

        retrieval_name = self.name_list[idx].split('_')[-1]

        motion, m_length, text_list = data["motion"], data["length"], data[
            "text"]

        text_data = random.choice(text_list)
        caption=text_data["caption"]
        sent_len=text_data['sent_len']
    
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if np.any(np.isnan(motion)):
            raise ValueError("nan in motion")

        return (
            caption,
            sent_len,
            motion,
            m_length,
            retrieval_name
        )








class Text2MotionDatasetV3_NPZ(Dataset):
  
    def __init__(
        self,
        mean,
        std,
        split_datapart,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        motion_dir,
        text_dir,
        input_format, 
        njoints, 
        tiny=False,
        debug=False,
        progress_bar=True,
        **kwargs,
    ):
   
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length

        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length

        data_dict = {}
        id_list = []

        self.id_list = id_list
        self.split = split_datapart
        if tiny or debug:
            self.lmdb_path = f''
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            self.lmdb_path = f''
            maxdata = 1e10
     
        self.to_tensor = True
        self.dtype_map = None or {}

        self.mean = torch.from_numpy(mean.reshape(2, 21, -1))
        self.std  = torch.from_numpy(std.reshape(2, 21, -1))
        
        self.nfeats = self.mean.reshape(-1).shape[0]
      
        env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False)
        with env.begin() as txn:
            raw = txn.get(b'__keys__')
            if raw is None:
                raise RuntimeError("LMDB ERROR")
            self.keys = json.loads(raw.decode('utf-8'))
        env.close()

        self.env = None
        self.txn = None

    def __getstate__(self):
        d = dict(self.__dict__)
        d['env'] = None
        d['txn'] = None
        return d

    def _lazy_init(self):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False,
                                 readahead=False, max_readers=4096)
            self.txn = self.env.begin(buffers=True)

    def __len__(self):
        return len(self.keys)

    def _maybe_to_tensor(self, name, value):
        if not self.to_tensor:
            return value
        if isinstance(value, np.ndarray):
            t = torch.from_numpy(value)

            if name in self.dtype_map:
                t = t.to(self.dtype_map[name])
            elif t.dtype == torch.float64:
                t = t.float()  
            return t
        return value 

    def __getitem__(self, idx):
        self._lazy_init()
        name = self.keys[idx]

        f = 'motion'
        key = f"{name}:{f}".encode("utf-8")
        buf = self.txn.get(key)
        if buf is None:
            raise KeyError(f"missing key: {name}:{f}")
        motion = torch.from_numpy(_load_value(bytes(buf)))
        motion = (motion-self.mean)/self.std
        motion = motion.reshape(motion.shape[0],-1)
        
        texts =[]
        for f in ['left_annotation','right_annotation','interaction_annotation']:
            key = f"{name}:{f}".encode("utf-8")
            buf = self.txn.get(key)
            if buf is None:
                raise KeyError(f"missing key: {name}:{f}")
            feature = (_load_value(bytes(buf)))
            N =len(feature)
            i = np.random.randint(low=0, high=N)
            texts.append(feature[i])
        caption = f'Left: {texts[0]}; Right: {texts[1]}; Interaction: {texts[2]}'
     
        return (
            caption,
            77,
            motion,
            60,
            name
        )



class Text2MotionDatasetV3(Dataset):
    
    
    def __init__(
        self,
        mean,
        std,
        split_datapart,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        motion_dir,
        text_dir,
        input_format, 
        njoints, 
        tiny=False,
        debug=False,
        progress_bar=True,
        **kwargs,
    ):
 
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length

        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length

        data_dict = {}
        id_list = []

        self.id_list = id_list
        self.split = split_datapart
        if tiny or debug:
            self.lmdb_path = f'{motion_dir}/test_can_pos_all_wotextfeat.npz'
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            self.lmdb_path = f'{motion_dir}/{self.split}_can_pos_all_wotextfeat.npz'
            maxdata = 1e10
     
        self.to_tensor = True
        self.dtype_map = None or {}

        
        self.mean = torch.from_numpy(mean.reshape(2, 21, -1))
        self.std  = torch.from_numpy(std.reshape(2, 21, -1))
        
        self.nfeats = self.mean.reshape(-1).shape[0]
     

    
        self.data = dict(np.load(self.lmdb_path,allow_pickle=True))
        self.keys = list(self.data.keys())
        

        self.env = None
        self.txn = None



    def __len__(self):
        return len(self.keys)

    def _maybe_to_tensor(self, name, value):
        if not self.to_tensor:
            return value
        if isinstance(value, np.ndarray):
            t = torch.from_numpy(value)

            if name in self.dtype_map:
                t = t.to(self.dtype_map[name])
            elif t.dtype == torch.float64:
                t = t.float()  
            return t
        return value  

    def __getitem__(self, idx):
        # self._lazy_init()
        name = self.keys[idx]
        dct = self.data[name].item()
        motion = (dct['motion'])
     
        motion = (torch.from_numpy(motion)-self.mean)/self.std
        motion = motion.reshape(motion.shape[0],-1)
        
        texts =[]
        for f in ['left_annotation','right_annotation','interaction_annotation']:
         
            feature = dct[f]
            N =len(feature)
            i = np.random.randint(low=0, high=N)
            texts.append(feature[i])
        caption = f'<extra_id0> {texts[0]} <extra_id1> {texts[1]} <extra_id2> {texts[2]} <extra_id3>'
     
        return (
            caption,
            77,
            motion,
            60,
            name
        )
   