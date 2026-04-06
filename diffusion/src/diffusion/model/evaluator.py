from collections import OrderedDict
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset, SubsetRandomSampler
import numpy as np
from einops import rearrange
from typing import List
from scipy import linalg
from tqdm import tqdm

from ..logger_new import mylogger
from .actor import ACTORStyleEncoder
from ..config import DataLoaderConfig
from ..utils.mics import get_device
from ..data_loader.get_data import get_dataloader
from ..dist import get_world_size, gather_tensors, is_main_process, barrier, broadcast_tensor, get_rank

class Evaluator(object):
    def __init__(
        self,
        sample_model:nn.Module,
        train_dataset:Dataset,
        val_dataset:Dataset,
        dataloader_cfg:DataLoaderConfig,
        sample_fn,
        njoints:int, nfeats:int, sample_length:int,
        num_samples_on_train:int,
        num_samples_on_val:int,
        num_samples_per_condition:int,
    ):
        self.sample_model = sample_model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.raw_sample_fn = sample_fn
        self.dataloader_cfg = dataloader_cfg
        self.njoints = njoints
        self.nfeats = nfeats
        self.sample_length = sample_length
        self.num_samples_on_train = num_samples_on_train
        self.num_samples_on_val = num_samples_on_val
        self.num_samples_per_condition = num_samples_per_condition
        self.get_motion_encoder()
        self.get_text_encoder()
        self.get_text_tokenizer()

    def get_motion_encoder(self):
        self.motion_encoder = ACTORStyleEncoder(
            vae=True,
            latent_dim=256,
            ff_size=1024,
            num_layers=6,
            num_heads=4,
            dropout=0.1,
            activation='gelu',
            nfeats=126
        )

        # Updated checkpoint path: using self-trained TMR model
        checkpoint_path = "your_ckpt"

        full_checkpoint:OrderedDict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
        checkpoint = OrderedDict()
        for key, value in full_checkpoint.items():
            if key.startswith("motion_encoder."):
                new_key = key[len("motion_encoder."):]
                checkpoint[new_key] = value

        self.motion_encoder.load_state_dict(checkpoint, strict=True)
        self.motion_encoder.to(get_device())
        self.motion_encoder.eval()

    def get_text_encoder(self):
        self.text_encoder = ACTORStyleEncoder(
            vae=True,
            latent_dim=256,
            ff_size=1024,
            num_layers=6,
            num_heads=4,
            dropout=0.1,
            activation='gelu',
            nfeats=768  # DistilBERT embedding size
        )

        checkpoint_path = "your_ckpt"
        full_checkpoint = torch.load(checkpoint_path, map_location='cpu')['state_dict']
        checkpoint = OrderedDict()
        for key, value in full_checkpoint.items():
            if key.startswith("text_encoder."):
                new_key = key[len("text_encoder."):]
                checkpoint[new_key] = value

        self.text_encoder.load_state_dict(checkpoint, strict=True)
        self.text_encoder.to(get_device())
        self.text_encoder.eval()

    def get_text_tokenizer(self):
        from transformers import AutoTokenizer, AutoModel
        self.text_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.text_model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.text_model.to(get_device())
        self.text_model.eval()

    @staticmethod
    def calc_mu_and_cov(batch_data:np.ndarray):
        '''
        batch_data: (B, D)
        '''
        mu = np.mean(batch_data, axis=0) # (D,)
        cov = np.cov(batch_data, rowvar=False) # (D, D)
        return mu, cov

    @staticmethod
    def calc_frechet_distance(mu1:np.ndarray, cov1:np.ndarray, mu2:np.ndarray, cov2:np.ndarray, eps=1e-6):
        '''
        mu: (D,)
        cov: (D, D)
        '''
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        cov1 = np.atleast_2d(cov1)
        cov2 = np.atleast_2d(cov2)

        assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
        assert cov1.shape == cov2.shape, "Training and test covariance matrices have different shapes"

        diff = mu1 - mu2

        cov_mean, _ = linalg.sqrtm(cov1 @ cov2, disp=False)
        if not np.isfinite(cov_mean).all():
            msg = f"fid calculation produces singular product; adding {eps} to diagonal of cov estimates"
            mylogger.warning(msg)
            offset = np.eye(cov1.shape[0]) * eps
            covmean = linalg.sqrtm((cov1 + offset) @ (cov2 + offset), disp=False)

        if np.iscomplexobj(cov_mean):
            if not np.allclose(np.diagonal(cov_mean).imag, 0, atol=1e-3):
                m = np.max(cov_mean.imag)
                raise ValueError(f"Imaginary component {m}")
            cov_mean = cov_mean.real

        tr_covmean = np.trace(cov_mean)

        return (diff @ diff + np.trace(cov1) + np.trace(cov2) - 2 * tr_covmean)

    def get_clean_sample_fn(self):
        return lambda model, batch_size, model_kwargs: self.raw_sample_fn(
            model,
            (batch_size, self.njoints, self.nfeats, self.sample_length),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            device=get_device(),
            skip_timesteps=0,
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False
        )

    def collect_gt_sample_motion_pairs(self, dataloader:DataLoader, split:str):

        batch_size = dataloader.batch_size
        sample_fn = self.get_clean_sample_fn()

        gt_motions = []
        sample_motions = []
        masks = []
        text_embeds_list = []

        device = get_device()

        for x, model_kwargs in tqdm(dataloader, desc=f'RANK {get_rank()} | Collecting GT Sample Motion Pairs on {split.upper()}'):

            gt = x.clone()
            gt = rearrange(gt, 'b j f t -> b t (j f)')
            gt_motions.append(gt)
            masks.append(model_kwargs['y']['mask'].squeeze(1).squeeze(1))

            sample_result = sample_fn(self.sample_model, x.shape[0], model_kwargs)
            sample_result = rearrange(sample_result, 'b j f t -> b t (j f)')
            sample_motions.append(sample_result)

            # Process text to get embeddings
            texts = model_kwargs['y']['text']
            batch_text_embeds = []

            with torch.no_grad():
                # Handle dict of lists format (from treble collate)
                if isinstance(texts, dict) and 'left' in texts:
                    # texts is {'left': [str, ...], 'right': [str, ...], 'two_hands_relation': [str, ...]}
                    batch_size_local = len(texts['left'])
                    for i in range(batch_size_local):
                        text_str = (
                            f"The left hand behaves a motion of: {texts['left'][i]}. "
                            f"The right hand behaves a motion of: {texts['right'][i]}. "
                            f"And the relation between left and right hand is: {texts['two_hands_relation'][i]}."
                        )

                        # Tokenize
                        tokens = self.text_tokenizer(text_str, return_tensors='pt',
                                                   padding='max_length', max_length=512, truncation=True)
                        input_ids = tokens['input_ids'].to(device)
                        attention_mask = tokens['attention_mask'].to(device)

                        # Get text embeddings
                        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
                        text_token_embeds = outputs.last_hidden_state

                        # Encode with text encoder
                        text_embed = self.text_encoder(dict(
                            x=text_token_embeds,
                            mask=attention_mask.bool()
                        ))[:, 0]  # Take mu only

                        batch_text_embeds.append(text_embed)
                else:
                    # Handle list format (fallback for other dataset types)
                    for text in texts:
                        if isinstance(text, dict):
                            text_str = (
                                f"The left hand behaves a motion of: {text.get('left', '')}. "
                                f"The right hand behaves a motion of: {text.get('right', '')}. "
                                f"And the relation between left and right hand is: {text.get('two_hands_relation', '')}."
                            )
                        else:
                            text_str = str(text)

                        # Tokenize
                        tokens = self.text_tokenizer(text_str, return_tensors='pt',
                                                   padding='max_length', max_length=512, truncation=True)
                        input_ids = tokens['input_ids'].to(device)
                        attention_mask = tokens['attention_mask'].to(device)

                        # Get text embeddings
                        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
                        text_token_embeds = outputs.last_hidden_state

                        # Encode with text encoder
                        text_embed = self.text_encoder(dict(
                            x=text_token_embeds,
                            mask=attention_mask.bool()
                        ))[:, 0]  # Take mu only

                        batch_text_embeds.append(text_embed)

                # Stack batch embeddings
                batch_text_embeds = torch.cat(batch_text_embeds, dim=0)
                text_embeds_list.append(batch_text_embeds)

        return gt_motions, sample_motions, masks, text_embeds_list

    def collect_sample_motions_for_multimodality(self, dataloader:DataLoader, split:str):
        def multiple_data(data:torch.Tensor | list | dict):
            if isinstance(data, torch.Tensor):
                new_data = torch.stack([data] * self.num_samples_per_condition, dim=1)
                return new_data.reshape(-1, *data.shape[1:]) # (B*S_m, ...)
            elif isinstance(data, list):
                ret_data = []
                for d in data:
                    ret_data += [d] * self.num_samples_per_condition
                return ret_data
            elif isinstance(data, dict):
                # Handle nested dictionary (e.g., text: {left, right, two_hands_relation})
                ret_data = {}
                for k, v in data.items():
                    ret_data[k] = multiple_data(v)
                return ret_data
            else:
                # For other types, just replicate
                return data
        sample_fn = self.get_clean_sample_fn()

        sample_motions = []
        masks = []

        for x, model_kwargs in tqdm(dataloader, desc=f'RANK {get_rank()} | Collecting Sample Motions for Multimodality {split.upper()}'):
            for key, value in model_kwargs['y'].items():
                new_value = multiple_data(value)
                model_kwargs['y'][key] = new_value

            masks.append(model_kwargs['y']['mask'].squeeze(1).squeeze(1))

            sample_result = sample_fn(self.sample_model, x.shape[0] * self.num_samples_per_condition, model_kwargs)
            sample_result = rearrange(sample_result, 'b j f t -> b t (j f)')
            sample_motions.append(sample_result)

        return sample_motions, masks

    def get_motion_embeddings(self, motion_list:List[torch.Tensor], mask_list:List[torch.Tensor]):
        embeddings = []
        device = get_device()
        with torch.no_grad():
            for motion, mask in tqdm(zip(motion_list, mask_list), desc=f'RANK {get_rank()} | Getting Motion Embeddings'):
                embedding = self.motion_encoder(dict(
                    x=motion.to(device),
                    mask=mask.to(device)
                ))[:, 0] # Originally return mu and sigma, now only take mu

                embeddings.append(embedding)

        return torch.concat(embeddings, dim=0)

    def calc_fid_metric(self, gt_embeddings:np.ndarray, sample_embeddings:np.ndarray):
        gt_mu, gt_cov = self.calc_mu_and_cov(gt_embeddings)
        sample_mu, sample_cov = self.calc_mu_and_cov(sample_embeddings)
        return self.calc_frechet_distance(gt_mu, gt_cov, sample_mu, sample_cov)

    def calc_diversity_metric(self, embedding1:np.ndarray, embedding2:np.ndarray):
        '''
        embedding: (B, D)
        '''
        return np.mean(np.linalg.norm(embedding1 - embedding2, axis=1))

    def calc_cosine_similarity(self, embedding1:np.ndarray, embedding2:np.ndarray):
        '''
        Calculate average cosine similarity between paired embeddings
        embedding1: (B, D)
        embedding2: (B, D)
        Returns: average cosine similarity
        '''
        # Normalize embeddings
        embedding1_norm = embedding1 / (np.linalg.norm(embedding1, axis=1, keepdims=True) + 1e-8)
        embedding2_norm = embedding2 / (np.linalg.norm(embedding2, axis=1, keepdims=True) + 1e-8)

        # Compute cosine similarity for each pair
        cosine_similarities = np.sum(embedding1_norm * embedding2_norm, axis=1)
        return np.mean(cosine_similarities)

    def get_subdataset(self, split:str, for_multimodality:bool=False):
        if split == 'train':
            dataset = self.train_dataset
            num_samples = self.num_samples_on_train
        elif split == 'val':
            dataset = self.val_dataset
            num_samples = self.num_samples_on_val

        if for_multimodality:
            num_samples //= self.num_samples_per_condition

        with torch.no_grad():
            if is_main_process():
                all_indices = list(range(len(dataset)))
                random_indices = np.random.choice(all_indices, num_samples, replace=False)
                random_indices = torch.tensor(random_indices, dtype=torch.long, device=get_device())
            else:
                random_indices = torch.empty(num_samples, dtype=torch.long, device=get_device())

            broadcast_tensor(random_indices)
            barrier()

            random_indices = random_indices.detach().cpu().numpy()

        subset = Subset(dataset, indices=random_indices)
        return subset

    def get_dataloader_for_multimodality(self, dataset:Dataset):
        return get_dataloader(dataset, DataLoaderConfig(
            batch_size=self.dataloader_cfg.batch_size // self.num_samples_per_condition,
            num_workers=self.dataloader_cfg.num_workers,
            shuffle=self.dataloader_cfg.shuffle
        ))

    def evaluate(self, split:str):
        sub_dataset_plain = self.get_subdataset(split)
        dataloader_plain = get_dataloader(sub_dataset_plain, self.dataloader_cfg)

        gt_motions, sample_motions, masks, text_embeds_list = self.collect_gt_sample_motion_pairs(dataloader_plain, split)

        sub_dataset_for_multimodality = self.get_subdataset(split, for_multimodality=True)
        dataloader_for_multimodality = self.get_dataloader_for_multimodality(sub_dataset_for_multimodality)
        sample_motions_for_multimodality, masks_for_multimodality = self.collect_sample_motions_for_multimodality(dataloader_for_multimodality, split)

        gt_embeddings = self.get_motion_embeddings(gt_motions, masks)
        sample_embeddings = self.get_motion_embeddings(sample_motions, masks)

        # Concatenate text embeddings
        text_embeddings = torch.concat(text_embeds_list, dim=0)

        gt_embeddings = gather_tensors(gt_embeddings)
        sample_embeddings = gather_tensors(sample_embeddings)
        text_embeddings = gather_tensors(text_embeddings)

        sample_embeddings_for_multimodality = self.get_motion_embeddings(sample_motions_for_multimodality, masks_for_multimodality)
        sample_embeddings_for_multimodality = gather_tensors(sample_embeddings_for_multimodality)

        if not is_main_process():
            return None

        gt_embeddings = torch.concat(gt_embeddings, dim=0).detach().cpu().numpy()
        sample_embeddings = torch.concat(sample_embeddings, dim=0).detach().cpu().numpy()
        text_embeddings = torch.concat(text_embeddings, dim=0).detach().cpu().numpy()
        sample_embeddings_for_multimodality = torch.concat(sample_embeddings_for_multimodality, dim=0).detach().cpu().numpy()

        ret_dict = dict()
        ret_dict[f'{split}_fid'] = self.calc_fid_metric(gt_embeddings, sample_embeddings)

        # Compute FID between text and motion embeddings
        ret_dict[f'{split}_fid_text_gt'] = self.calc_fid_metric(text_embeddings, gt_embeddings)
        ret_dict[f'{split}_fid_text_gen'] = self.calc_fid_metric(text_embeddings, sample_embeddings)

        # Compute cosine similarity between paired text and motion embeddings
        ret_dict[f'{split}_cosine_sim_text_gt'] = self.calc_cosine_similarity(text_embeddings, gt_embeddings)
        ret_dict[f'{split}_cosine_sim_text_gen'] = self.calc_cosine_similarity(text_embeddings, sample_embeddings)

        if sample_embeddings.shape[0] % 2 == 1:
            sample_embeddings = sample_embeddings[:-1]

        sample_embeddings1, sample_embeddings2 = np.split(sample_embeddings, 2, axis=0)
        ret_dict[f'{split}_diversity'] = self.calc_diversity_metric(sample_embeddings1, sample_embeddings2)

        sample_embeddings_for_multimodality = sample_embeddings_for_multimodality.reshape(-1, self.num_samples_per_condition, *sample_embeddings_for_multimodality.shape[1:]) # (C, S_m * 2, D)
        sample_embeddings1_for_multimodality, sample_embeddings2_for_multimodality = np.split(sample_embeddings_for_multimodality, 2, axis=1) # (C, S_m, D), (C, S_m, D)
        sample_embeddings1_for_multimodality = sample_embeddings1_for_multimodality.reshape(-1, sample_embeddings1_for_multimodality.shape[-1])
        sample_embeddings2_for_multimodality = sample_embeddings2_for_multimodality.reshape(-1, sample_embeddings2_for_multimodality.shape[-1])
        ret_dict[f'{split}_multimodality'] = self.calc_diversity_metric(sample_embeddings1_for_multimodality, sample_embeddings2_for_multimodality)

        return ret_dict
