import os

import numpy as np
import torch
from tqdm import tqdm

from utils.motion_process import recover_from_local_position


@torch.no_grad()
def compute_perplexity(codebook_size, code_idx):
    code_onehot = torch.zeros(codebook_size, code_idx.shape[0], device=code_idx.device)
    code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

    code_count = code_onehot.sum(dim=-1)
    prob = code_count / torch.sum(code_count)
    perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
    activate = torch.sum(code_count > 0).float() / codebook_size
    return perplexity, activate


def calculate_mpjpe(gt_joints, pred_joints):
    """
    gt_joints: num_poses x num_joints(22) x 3
    pred_joints: num_poses x num_joints(22) x 3
    """
    assert gt_joints.shape == pred_joints.shape, f"GT shape: {gt_joints.shape}, pred shape: {pred_joints.shape}"

    N, J, D = gt_joints.shape
    gt_joints = gt_joints.reshape(N, 2, -1, D)
    pred_joints = pred_joints.reshape(N, 2, -1, D)

    gt_joints_l = gt_joints[:, 0]
    pred_joints_l = pred_joints[:, 0]
    pelvis = gt_joints_l[:, [0]].mean(1)
    gt_joints_l = gt_joints_l - torch.unsqueeze(pelvis, dim=1)
    pelvis = pred_joints_l[:, [0]].mean(1)
    pred_joints_l = pred_joints_l - torch.unsqueeze(pelvis, dim=1)

    mpjpe = torch.linalg.norm(pred_joints_l - gt_joints_l, dim=-1)
    mpjpe_seq_l = mpjpe.mean(-1)

    gt_joints_l = gt_joints[:, 1]
    pred_joints_l = pred_joints[:, 1]
    pelvis = gt_joints_l[:, [0]].mean(1)
    gt_joints_l = gt_joints_l - torch.unsqueeze(pelvis, dim=1)
    pelvis = pred_joints_l[:, [0]].mean(1)
    pred_joints_l = pred_joints_l - torch.unsqueeze(pelvis, dim=1)

    mpjpe = torch.linalg.norm(pred_joints_l - gt_joints_l, dim=-1)
    mpjpe_seq_r = mpjpe.mean(-1)
    mpjpe_seq = (mpjpe_seq_l + mpjpe_seq_r) / 2
    return mpjpe_seq


def calculate_acceleration(gt_joints, pred_joints):
    """
    gt_joints: num_poses x num_joints(22) x 3
    pred_joints: num_poses x num_joints(22) x 3
    """
    assert gt_joints.shape == pred_joints.shape, f"GT shape: {gt_joints.shape}, pred shape: {pred_joints.shape}"

    pelvis = gt_joints[:, [0]].mean(1)
    gt_joints = gt_joints - torch.unsqueeze(pelvis, dim=1)
    pelvis = pred_joints[:, [0]].mean(1)
    pred_joints = pred_joints - torch.unsqueeze(pelvis, dim=1)

    pred_velocity = pred_joints[1:] - pred_joints[:-1]
    pred_acceleration = pred_velocity[1:] - pred_velocity[:-1]
    pred_mean_acceleration_seq = torch.linalg.norm(pred_acceleration, dim=-1).mean(-1)
    pred_max_acceleration_seq = torch.linalg.norm(pred_acceleration, dim=-1).max(-1)[0]

    gt_velocity = gt_joints[1:] - gt_joints[:-1]
    gt_acceleration = gt_velocity[1:] - gt_velocity[:-1]
    gt_mean_acceleration_seq = torch.linalg.norm(gt_acceleration, dim=-1).mean(-1)
    gt_max_acceleration_seq = torch.linalg.norm(gt_acceleration, dim=-1).max(-1)[0]

    return pred_mean_acceleration_seq, pred_max_acceleration_seq, gt_mean_acceleration_seq, gt_max_acceleration_seq


@torch.no_grad()
def evaluation_vqvae_motionmillion(out_dir, train_loader, val_loader, net, logger, writer, nb_iter, best_mpjpe, comp_device, codebook_size, draw=True, save=True, savenpy=False, accelerator=None, cal_acceleration=False):
    net.eval()

    mpjpe = torch.tensor(0.0, device=comp_device)
    gpe = torch.tensor(0.0, device=comp_device)
    rpe1 = torch.tensor(0.0, device=comp_device)
    rpe2 = torch.tensor(0.0, device=comp_device)
    if cal_acceleration:
        pred_mean_acceleration_seq = torch.tensor(0.0, device=comp_device)
        pred_max_acceleration_seq = torch.tensor(0.0, device=comp_device)
        gt_mean_acceleration_seq = torch.tensor(0.0, device=comp_device)
        gt_max_acceleration_seq = torch.tensor(0.0, device=comp_device)

    nb_sample = torch.tensor(0, device=comp_device)

    motion_indices = []

    for batch in tqdm(val_loader):
        motion, m_length, name = batch

        motion = motion.to(comp_device)

        bs, seq = motion.shape[0], motion.shape[1]

        num_joints = 20

        pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).to(comp_device)

        for i in range(bs):
            pose = val_loader.dataset.inv_transform(motion[i:i+1, :m_length[i], :].detach().cpu().numpy())
            pose_xyz = torch.from_numpy(recover_from_local_position(pose.squeeze(), num_joints)).float().to(comp_device).unsqueeze(0)

            pose = train_loader.dataset.transform(pose)

            pred_pose, _, perplexity, activate, indices = net(torch.from_numpy(pose).to(comp_device))
            motion_indices.append(indices.squeeze().cpu())

            pred_denorm = train_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
            pred_xyz = torch.from_numpy(recover_from_local_position(pred_denorm.squeeze(), num_joints)).float().to(comp_device).unsqueeze(0)
            mpjpe += torch.mean(calculate_mpjpe(pose_xyz[:].squeeze(), pred_xyz[:].squeeze()))
            gpe += torch.mean(torch.linalg.norm(pose_xyz[:, :] - pred_xyz[:, :], dim=-1))
            _, T, K, D = pose_xyz.shape
            rpe1 += torch.mean(torch.linalg.norm(pose_xyz.reshape(T, 2, -1, 3)[:, :, 0] - pred_xyz.reshape(T, 2, -1, 3)[:, :, 0], dim=-1))
            rpe2 += torch.mean(torch.linalg.norm(pose_xyz.reshape(T, 2, -1, 3)[:, :, 1] - pred_xyz.reshape(T, 2, -1, 3)[:, :, 1], dim=-1))

            if cal_acceleration:
                pred_mean_acc, pred_max_acc, gt_mean_acc, gt_max_acc = calculate_acceleration(pose_xyz[:].squeeze(), pred_xyz[:].squeeze())
                pred_mean_acceleration_seq += torch.mean(pred_mean_acc)
                pred_max_acceleration_seq += torch.max(pred_max_acc)
                gt_mean_acceleration_seq += torch.mean(gt_mean_acc)
                gt_max_acceleration_seq += torch.max(gt_max_acc)

            if savenpy:
                np.save(os.path.join(out_dir, name[i]+'_gt.npy'), pose_xyz[:, :m_length[i]].cpu().numpy())
                np.save(os.path.join(out_dir, name[i]+'_pred.npy'), pred_xyz.detach().cpu().numpy())

            pred_pose = val_loader.dataset.transform(train_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy()))
            pred_pose_eval[i:i+1, :m_length[i], :] = torch.from_numpy(pred_pose).to(comp_device)

        nb_sample = nb_sample + bs

    if accelerator is None or accelerator.is_main_process:
        motion_indices = [motion_index.flatten() for motion_index in motion_indices]
        all_motion_indices = torch.cat(motion_indices)

        perplexity, activate = compute_perplexity(codebook_size, all_motion_indices.reshape(-1).to(torch.int64))
        mpjpe = mpjpe / nb_sample
        gpe = gpe / nb_sample
        rpe1 = rpe1 / nb_sample
        rpe2 = rpe2 / nb_sample
        if cal_acceleration:
            pred_mean_acceleration_seq = pred_mean_acceleration_seq / nb_sample
            pred_max_acceleration_seq = pred_max_acceleration_seq / nb_sample
            gt_mean_acceleration_seq = gt_mean_acceleration_seq / nb_sample
            gt_max_acceleration_seq = gt_max_acceleration_seq / nb_sample
        msg = f"--> \t Eva. Iter {nb_iter} :, MPJPE. {mpjpe:.4f} PPL. {perplexity} Activate. {activate:.4f}, GPE. {gpe:.4f}"
        if cal_acceleration:
            msg += f" Pred Mean Accel. {pred_mean_acceleration_seq:.4f} Pred Max Accel. {pred_max_acceleration_seq:.4f} GT Mean Accel. {gt_mean_acceleration_seq:.4f} GT Max Accel. {gt_max_acceleration_seq:.4f}"
        logger.info(msg)

    if accelerator.is_main_process:
        dict2 = {}
        dict2['Test/PPL'] = perplexity.item()
        dict2['Test/activate'] = activate.item()
        dict2['Test/MPJPE'] = mpjpe.item()
        dict2['Test/GPE'] = gpe.item()
        dict2['Test/RPE1'] = rpe1.item()
        dict2['Test/RPE2'] = rpe2.item()
        if cal_acceleration:
            dict2['Test/Pred Mean Accel'] = pred_mean_acceleration_seq.item()
            dict2['Test/Pred Max Accel'] = pred_max_acceleration_seq.item()
            dict2['Test/GT Mean Accel'] = gt_mean_acceleration_seq.item()
            dict2['Test/GT Max Accel'] = gt_max_acceleration_seq.item()

        accelerator.log(dict2, step=nb_iter)

    if accelerator is None or accelerator.is_main_process:
        if mpjpe < best_mpjpe:
            msg = f"--> --> \t MPJPE Improved from {best_mpjpe:.5f} to {mpjpe:.5f} !!!"
            logger.info(msg)
            best_mpjpe = mpjpe
            if save:
                torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_best_mpjpe.pth'))

    net.train()
    return best_mpjpe, writer, logger
