import torch
import torch.nn as nn
from bps_torch.bps import bps_torch
from pytorch3d.transforms import quaternion_to_matrix

class ReConsLoss(nn.Module):
    def __init__(self, recons_loss, nb_joints):
        super(ReConsLoss, self).__init__()
        
        if recons_loss == 'l1': 
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == 'l2' : 
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == 'l1_smooth' : 
            self.Loss = torch.nn.SmoothL1Loss()
        
        # 4 global motion associated to root
        # 12 local motion (3 local xyz, 3 vel xyz, 6 rot6d)
        # 3 global vel xyz
        # 4 foot contact
        self.nb_joints = nb_joints
        self.motion_dim = 438 # (nb_joints - 1) * 12 + 4 + 3 + 4
        
    def forward(self, motion_pred, motion_gt) : 
        loss = self.Loss(motion_pred[..., : self.motion_dim], motion_gt[..., :self.motion_dim])
        return loss
    
    def forward_vel(self, motion_pred, motion_gt) : 
        # loss = self.Loss(motion_pred[..., 4 : (self.nb_joints - 1) * 3 + 4], motion_gt[..., 4 : (self.nb_joints - 1) * 3 + 4])
        loss = self.Loss(motion_pred[..., :], motion_gt[..., :])
        return loss
    
    def forward_acc(self, motion_pred, motion_gt) : 
        B, T, D = motion_pred.shape
        
        pred_vel = motion_pred[:,1:, 8 : 8+3*self.nb_joints]-motion_pred[:, :-1, 8 : 8+3*self.nb_joints]
        gt_vel = motion_gt[:,1:, 8 : 8+3*self.nb_joints]-motion_gt[:, :-1, 8 : 8+3*self.nb_joints]
        
        loss = self.Loss(pred_vel, gt_vel)
        
        return loss
    
    def forward_acc_vel(self, motion_pred, motion_gt) : 
        B, T, D = motion_pred.shape
        
        pred_vel = motion_pred[:,1:, 8 : 8+3*self.nb_joints]-motion_pred[:, :-1, 8 : 8+3*self.nb_joints]
        pred_acc = pred_vel[:,1:, :]-pred_vel[:, :-1, :]
        
        gt_vel = motion_gt[:,1:, 8 : 8+3*self.nb_joints]-motion_gt[:, :-1, 8 : 8+3*self.nb_joints]
        gt_acc = gt_vel[:,1:, :]-gt_vel[:, :-1, :]
        
        loss = self.Loss(pred_acc, gt_acc)
        
        return loss
    
    
    def forward_root(self, motion_pred, motion_gt) : 
        loss = self.Loss(motion_pred[..., :8], motion_gt[..., :8])
        return loss

def cont6d_to_matrix(cont6d):
    assert cont6d.shape[-1] == 6, "The last dimension must be 6"
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]

    x = x_raw / torch.norm(x_raw, dim=-1, keepdim=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.norm(z, dim=-1, keepdim=True)

    y = torch.cross(z, x, dim=-1)

    x = x[..., None]
    y = y[..., None]
    z = z[..., None]

    mat = torch.cat([x, y, z], dim=-1)
    return mat
class Loss_witer(nn.Module):
    def __init__(self, recons_loss, nb_joints,input_dim):
        super(Loss_witer, self).__init__()
        
        if recons_loss == 'l1': 
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == 'l2' : 
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == 'l1_smooth' : 
            self.Loss = torch.nn.SmoothL1Loss()
        self.bps = bps_torch(bps_type='random_uniform',
                n_bps_points=1024,
                radius=1.,
                n_dims=3,
                custom_basis=None)
        
        # 4 global motion associated to root
        # 12 local motion (3 local xyz, 3 vel xyz, 6 rot6d)
        # 3 global vel xyz
        # 4 foot contact
        self.nb_joints = nb_joints
        self.motion_dim = input_dim # (nb_joints - 1) * 12 + 4 + 3 + 4
        if self.motion_dim == 438:
            self.main_component = list(range(6)) + list(range(6,12))+ list(range(6+16*6,12+16*6))+list(range(6+2*16*6,6+2*16*6+2*20*3))
            self.redundant_component = list(range(12,6+16*6))+ list(range(12+16*6,6+16*6*2))+list(range(6+2*16*6+2*20*3,438))
            self.pos_index = list(range(6+2*16*6,6+2*16*6+2*20*3))
            self.rot_index = list(range(6)) +list(range(6,12))+ list(range(6+16*6,12+16*6))
            self.r_rot_index =  list(range(6,12))+ list(range(6+16*6,12+16*6))
        elif self.motion_dim in [168,288]:
            self.main_component = list(range(6)) + list(range(6,18))+ list(range(18+30,18+30+2*20*3*2))
            self.redundant_component = list(range(18,48))
            self.pos_index = list(range(48,48+2*20*3))
            self.rot_index = list(range(6)) +list(range(6,18))#+ list(range(6+16*6,12+16*6))
            self.r_rot_index = list(range(6,18))
            self.root_pos_index = list(range(6))
            self.related_index = list(range(3,6))
        elif self.motion_dim in [164,134,284]:
            self.main_component = list(range(6)) + list(range(6,14))+ list(range(18+30,18+30+2*20*3*2))
            self.redundant_component = list(range(14,44))
            self.pos_index = list(range(14,14+2*20*3))
            self.rot_index = list(range(6)) +list(range(6,14))#+ list(range(6+16*6,12+16*6))
            self.r_rot_index = list(range(6,14))
            self.root_pos_index = list(range(6))
            # self.related_index = list(range(6,9))
        elif self.motion_dim in [291]:
            self.main_component = list(range(6)) + list(range(6,18))+ list(range(18+30,18+30+2*20*3*2))
            self.redundant_component = list(range(18,48))
            self.pos_index = list(range(51,51+2*20*3))
            self.rot_index = list(range(6)) +list(range(9,18+3))#+ list(range(6+16*6,12+16*6))
            self.r_rot_index = list(range(6+3,18+3))
            self.root_pos_index = list(range(9))
            self.related_index = list(range(6,9))
        
        self.kinematic_chain = [[0,13,14,15,16],[0,1,2,3,17],[0,4,5,6,18],[0,10,11,12,19],[0,7,8,9,20]]

    def forward(self, motion_pred, motion_gt):
        loss = self.Loss(motion_pred[:], motion_gt[:])
        # if self.motion_dim == 258:
        #     loss = self.Loss(motion_pred[:], motion_gt[:])
        # else: 
        #     loss = self.Loss(motion_pred[...,self.main_component], motion_gt[...,self.main_component]) + 0.3*self.Loss(motion_pred[...,self.redundant_component], motion_gt[...,self.redundant_component])
        return loss
    
    def forward_vel(self, motion_pred, motion_gt) : 
        # loss = self.Loss(motion_pred[..., 4 : (self.nb_joints - 1) * 3 + 4], motion_gt[..., 4 : (self.nb_joints - 1) * 3 + 4])
        if self.motion_dim == 258:
            loss = self.Loss(motion_pred[..., 18:18+2*20*3], motion_gt[..., 18:18+2*20*3])
        else:
            loss = self.Loss(motion_pred[..., self.pos_index], motion_gt[..., self.pos_index])
        return loss
    def forward_rel_vel_unnorm(self, motion_pred, motion_gt) : 
        vel_A = motion_pred[...,:3] 
        rel_B = motion_pred[...,self.related_index] 
        pred_vb = rel_B[:,1:] - rel_B[:,:-1] + vel_A[:,:-1]
        
        vel_Agt = motion_gt[...,:3] 
        rel_Bgt = motion_gt[...,self.related_index] 
        pred_vbgt = rel_Bgt[:,1:] - rel_Bgt[:,:-1] + vel_Agt[:,:-1]
        return self.Loss(pred_vb,pred_vbgt)
    def forward_rel_vel(self, motion_pred, motion_gt) : 
        rel1 = motion_pred[...,self.related_index] 
        rel2 = motion_gt[...,self.related_index] 
        # pred_vb = rel_B[:,1:] - rel_B[:,:-1] + vel_A[:,:-1]
        
        # vel_Agt = motion_gt[...,:3] 
        # rel_Bgt = motion_gt[...,self.related_index] 
        # pred_vbgt = rel_Bgt[:,1:] - rel_Bgt[:,:-1] + vel_Agt[:,:-1]
        return self.Loss(rel1,rel2)
        # loss = self.Loss(motion_pred[..., 4 : (self.nb_joints - 1) * 3 + 4], motion_gt[..., 4 : (self.nb_joints - 1) * 3 + 4])
        # if self.motion_dim == 258:
        #     loss = self.Loss(motion_pred[..., 18:18+2*20*3], motion_gt[..., 18:18+2*20*3])
        # else:
        #     loss = self.Loss(motion_pred[..., self.pos_index], motion_gt[..., self.pos_index])
        # return loss
    def forward_vel_unnorm(self, motion_pred, motion_gt) : 
        # loss = self.Loss(motion_pred[..., 4 : (self.nb_joints - 1) * 3 + 4], motion_gt[..., 4 : (self.nb_joints - 1) * 3 + 4])
        if self.motion_dim == 258:
            loss = self.Loss(motion_pred[..., 18:18+2*20*3], motion_gt[..., 18:18+2*20*3])
        else:
            loss = self.Loss(motion_pred[..., self.pos_index], motion_gt[..., self.pos_index])
        return loss
    
    def forward_root(self, motion_pred, motion_gt):
        if self.motion_dim == 258:
            loss = self.Loss(motion_pred[..., 18:18+2*20*3], motion_gt[..., 18:18+2*20*3])
        else:
            loss = self.Loss(motion_pred[..., self.root_pos_index], motion_gt[..., self.root_pos_index])
        return loss
    def forward_root_first(self, motion_pred, motion_gt):
        loss = self.Loss(motion_pred[:,0, 6:9], motion_gt[:,0, 6:9])
        return loss
        # if self.motion_dim == 258:
        #     loss = self.Loss(motion_pred[..., 18:18+2*20*3], motion_gt[..., 18:18+2*20*3])
        # else:
        #     loss = self.Loss(motion_pred[..., self.root_pos_index], motion_gt[..., self.root_pos_index])
        # return loss
    # def forward_root_rot_l1(self, motion_pred, motion_gt):
    #     if self.motion_dim == 258:
    #         loss = self.Loss(motion_pred[..., 18:18+2*20*3], motion_gt[..., 18:18+2*20*3])
    #     else:
    #         loss = self.Loss(motion_pred[..., self.root_pos_index], motion_gt[..., self.root_pos_index])
    def forward_root_rot_l1s(self, motion_pred, motion_gt):
        return self.Loss(motion_pred[..., self.r_rot_index],motion_gt[..., self.r_rot_index])
    def forward_root_rot(self, motion_pred, motion_gt): #forward_root_rot_l1s
        
        B,T,D = motion_pred.shape
        if self.motion_dim in [168,288,291]:
            rot1 = motion_pred[..., self.r_rot_index].reshape(B,T,2,6)
            rot2 = motion_gt[..., self.r_rot_index].reshape(B,T,2,6)
            # loss_l1_s = self.Loss(rot1,rot2)
            r1 = cont6d_to_matrix(rot1).reshape(-1,3,3)
            r2 = cont6d_to_matrix(rot2).reshape(-1,3,3)
            loss_absolute = self.calc_loss_geo(r1,r2)
            
            r1_v = torch.matmul(r1[1:],r1[:-1].permute(0,2,1))
            r2_v = torch.matmul(r2[1:],r2[:-1].permute(0,2,1))
            loss_absolute_v = self.calc_loss_geo(r1_v,r2_v)
            # local_position = motion_pred[...,self.pos_index].reshape(-1,20,3)
            # global_position_woroot= torch.matmul(local_position,r1.permute(0,2,1))
            # global_position_woroot= torch.matmul(local_position,r1.permute(0,2,1))
            
            return loss_absolute,loss_absolute_v#,loss_l1_s
        elif self.motion_dim in [164,134,284]:
            B,T,D = motion_pred.shape
            rot1 = motion_pred[..., self.r_rot_index].reshape(B,T,2,4)
            rot2 = motion_gt[..., self.r_rot_index].reshape(B,T,2,4)
            # loss_l1_s = self.Loss(rot1,rot2)
            r1 = quaternion_to_matrix(rot1.float()).reshape(-1,3,3)
            r2= quaternion_to_matrix(rot2.float()).reshape(-1,3,3)
            loss_absolute = self.calc_loss_geo(r1,r2)
            r1_v = torch.matmul(r1[1:],r1[:-1].permute(0,2,1))
            r2_v = torch.matmul(r2[1:],r2[:-1].permute(0,2,1))
            loss_absolute_v = self.calc_loss_geo(r1_v,r2_v)
            
            return loss_absolute,loss_absolute_v#,loss_l1_s
            # return self.calc_loss_geo(r1,r2)
        # if self.motion_dim == 258:
        #     loss = self.Loss(motion_pred[..., 18:18+2*20*3], motion_gt[..., 18:18+2*20*3])
        # else:
        #     loss = self.Loss(motion_pred[..., self.rot_index], motion_gt[..., self.rot_index])
            
        # return loss
    
    def calc_loss_geo(self, pred_m, gt_m, eps=1e-7):
        # if self.dataset_name == "interhuman":
        #     pred_rot = pred_rot.reshape(pred_rot.shape[0], pred_rot.shape[1], -1, 6)
        #     gt_rot = gt_rot.reshape(gt_rot.shape[0], gt_rot.shape[1], -1, 6)


        # pred_m = cont6d_to_matrix(pred_rot).reshape(-1,3,3)
        # gt_m = cont6d_to_matrix(gt_rot).reshape(-1,3,3)

        m = torch.bmm(gt_m, pred_m.transpose(1,2)) #batch*3*3
        
        cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2        
        theta = torch.acos(torch.clamp(cos, -1+eps, 1-eps))

        return torch.mean(theta)
    def calc_bone_lengths(self, motion):
        # if self.dataset_name == 'interhuman':
        #     motion_pos = motion[..., :self.joints_num*3]
        #     motion_pos = motion_pos.reshape(motion_pos.shape[0], motion_pos.shape[1], self.joints_num, 3)
        # elif self.dataset_name == 'interx':
        motion_pos = motion
        bones = []
        for chain in self.kinematic_chain:
            for i, joint in enumerate(chain[:-1]):
                bone = (motion_pos[..., chain[i], :] - motion_pos[..., chain[i + 1], :]).norm(dim=-1, keepdim=True)  # [B,T,P,1]
                bones.append(bone)
        return torch.cat(bones, dim=-1)
        
    
    def forward_acc(self, motion_pred, motion_gt) : 
        B, T, D = motion_pred.shape
        
        pred_vel = motion_pred[:,1:, 8 : 8+3*self.nb_joints]-motion_pred[:, :-1, 8 : 8+3*self.nb_joints]
        gt_vel = motion_gt[:,1:, 8 : 8+3*self.nb_joints]-motion_gt[:, :-1, 8 : 8+3*self.nb_joints]
        
        loss = self.Loss(pred_vel, gt_vel)
        
        return loss
    def forward_bl(self, motion_pred, motion_gt):
        B,T,D = motion_pred.shape
        if self.motion_dim in [168,288,291]:
            local_pos = motion_pred[:,:,self.pos_index].reshape(B,T,2,20,3)
            local_pos_gt = motion_gt[:,:,self.pos_index].reshape(B,T,2,20,3)
            local_pos = torch.cat([torch.zeros_like(local_pos[:,:,:,:1]).to(local_pos.device),local_pos],3)
            local_pos_gt = torch.cat([torch.zeros_like(local_pos_gt[:,:,:,:1]).to(local_pos_gt.device),local_pos_gt],3)
            return self.Loss(self.calc_bone_lengths(local_pos), self.calc_bone_lengths(local_pos_gt))
        elif self.motion_dim in [164,134,284]:
            local_pos = motion_pred[:,:,14:14+2*20*3].reshape(B,T,2,20,3)
            local_pos_gt = motion_gt[:,:,14:14+2*20*3].reshape(B,T,2,20,3)
            local_pos = torch.cat([torch.zeros_like(local_pos[:,:,:,:1]).to(local_pos.device),local_pos],3)
            local_pos_gt = torch.cat([torch.zeros_like(local_pos_gt[:,:,:,:1]).to(local_pos_gt.device),local_pos_gt],3)
            return self.Loss(self.calc_bone_lengths(local_pos), self.calc_bone_lengths(local_pos_gt))
            
    
    def forward_vt(self, motion_pred, motion_gt): 
        loss = self.Loss(motion_pred[...,:3],  motion_gt[...,:3])
        return loss
        
    def forward_tt(self, motion_pred, motion_gt):
        loss = self.Loss(motion_pred[...,3:6],  motion_gt[...,3:6])
        return loss
    
    def forward_root_rot_vel(self, motion_pred, motion_gt) : 
        # loss = self.Loss(motion_pred[..., 4 : (self.nb_joints - 1) * 3 + 4], motion_gt[..., 4 : (self.nb_joints - 1) * 3 + 4])
        B, T, D = motion_pred.shape
        
        if self.motion_dim == 258:
            pred_vel = motion_pred[:,1:, 6:18]-motion_pred[:, :-1,  6:18]
            pred_acc = pred_vel[:,1:, :]-pred_vel[:, :-1, :]
            
            gt_vel = motion_gt[:,1:,  6:18]-motion_gt[:, :-1,  6:18]
            gt_acc = gt_vel[:,1:, :]-gt_vel[:, :-1, :]
        else:
            pred_vel = motion_pred[:,1:, self.r_rot_index]-motion_pred[:, :-1, self.r_rot_index]
            pred_acc = pred_vel[:,1:, :]-pred_vel[:, :-1, :]
            
            gt_vel = motion_gt[:,1:,  self.r_rot_index]-motion_gt[:, :-1,  self.r_rot_index]
            gt_acc = gt_vel[:,1:, :]-gt_vel[:, :-1, :]
            
        loss = self.Loss(pred_acc, gt_acc)
        return loss
    
    def forward_acc_vel(self, motion_pred, motion_gt) : 
        B, T, D = motion_pred.shape
        
        if self.motion_dim == 258:
            pred_vel = motion_pred[:,1:, 18:18+2*20*3]-motion_pred[:, :-1, 18:18+2*20*3]
            pred_acc = pred_vel[:,1:, :]-pred_vel[:, :-1, :]
            
            gt_vel = motion_gt[:,1:, 18:18+2*20*3]-motion_gt[:, :-1, 18:18+2*20*3]
            gt_acc = gt_vel[:,1:, :]-gt_vel[:, :-1, :]
        else:
            pred_vel = motion_pred[:,1:, self.pos_index]-motion_pred[:, :-1,  self.pos_index]
            pred_acc = pred_vel[:,1:, :]-pred_vel[:, :-1, :]
            
            gt_vel = motion_gt[:,1:,  self.pos_index]-motion_gt[:, :-1,  self.pos_index]
            gt_acc = gt_vel[:,1:, :]-gt_vel[:, :-1, :]
            
        loss = self.Loss(pred_acc, gt_acc)
        
        return loss
    
      # print(fuck)3+3+2*16*6+2*20*3+2*20*3
        # m2 = torch.cat([vel_A.reshape(60,-1),relative_B_pos.reshape(60,-1),rot6d.reshape(60,-1),local.reshape(60,-1),veloicty.reshape(60,-1)],-1)
    