import torch
from utils.quaternion import quaternion_to_cont6d, qrot, qinv
from pytorch3d.transforms import rotation_6d_to_matrix,quaternion_to_matrix
# from utils.face_z_align_util import rotation_6d_to_matrix, matrix_to_axis_angle
import numpy as np


def accumulate_rotations(relative_rotations):
    R_total = [relative_rotations[0]]
    for R_rel in relative_rotations[1:]:
        R_total.append(np.matmul(R_rel, R_total[-1]))
    
    return np.array(R_total)


def rotations_matrix_to_smpl85(rotations_matrix, translation):
    nfrm, njoint, _, _ = rotations_matrix.shape
    axis_angle = matrix_to_axis_angle(torch.from_numpy(rotations_matrix)).numpy().reshape(nfrm, -1)
    smpl_85 = np.concatenate([axis_angle, np.zeros((nfrm, 6)), translation, np.zeros((nfrm, 10))], axis=-1)
    return smpl_85



def recover_from_local_position(final_x, njoint=20):
    # take positions_no_heading: local position on xz ori, no heading
    # velocities_root_xy_no_heading: to recover translation
    # global_heading_diff_rot: to recover root rotation
    D = final_x.shape[-1]
    if D==258:
        nfrm, _ = final_x.shape
        positions_no_heading = final_x[:,18:18+2*20*3].reshape(nfrm, 2,-1, 3) # frames, njoints * 3
        # velocities_root_xy_no_heading = final_x[:,:2] # frames, 2
        global_heading_rot = final_x[:,6:18].reshape(nfrm, -1, 6) # frames, 6
        global_heading_rot = rotation_6d_to_matrix(torch.from_numpy(global_heading_rot)).numpy()
        # recover global heading
        # global_heading_rot = accumulate_rotations(rotation_6d_to_matrix(torch.from_numpy(global_heading_diff_rot)).numpy())
        inv_global_heading_rot = np.transpose(global_heading_rot, (0, 1,3, 2)) 
        # add global heading to position
        positions_with_heading = np.matmul(np.repeat(inv_global_heading_rot[:,:,None], 20, axis=2), positions_no_heading[...,None]).squeeze(-1)
        # print(positions_with_heading.shape) # 60,2,20,3
        # recover root translation
        # add heading to velocities_root_xy_no_heading

        # velocities_root_xyz_no_heading = np.zeros((velocities_root_xy_no_heading.shape[0], 3))
        # velocities_root_xyz_no_heading[:, 0] = velocities_root_xy_no_heading[:, 0]
        # velocities_root_xyz_no_heading[:, 2] = velocities_root_xy_no_heading[:, 1]
        # velocities_root_xyz_no_heading[1:, :] = np.matmul(inv_global_heading_rot[:-1], velocities_root_xyz_no_heading[1:, :,None]).squeeze(-1)
            #   relative_B_pos = position[:,0,0]-position[:,1,0]
            # vel_A = position[1:,1,0]-position[:1,1,0]
            # vel_A = torch.cat([vel_A,torch.zeros_like(vel_A[:1])],0) # pos,A,pos_B Translation,
        vel_A = final_x[:,:3] # right
        vel_B = final_x[:,3:6]
        root_right = np.cumsum(vel_A,axis=0)
        root_left = root_right+vel_B
        # print(root_left.shape,root_right.shape)
        
        # root_translation = np.cumsum(velocities_root_xyz_no_heading, axis=0)


        # add root translation
        positions_with_heading[:, 0, :] += root_left.reshape(nfrm,1,3)
        positions_with_heading[:, 1, :] += root_right.reshape(nfrm,1,3)
    elif D in [288,168]:
        nfrm, _ = final_x.shape
        positions_no_heading = final_x[:,18+30:18+30+2*20*3].reshape(nfrm, 2,-1, 3) # frames, njoints * 3
        # global_heading_rot_6d = final_x[:,6:18].reshape(nfrm, 2, 6)
        # global_heading_rot = rotation_6d_to_matrix(torch.from_numpy(global_heading_rot_6d)).numpy()  # R # N,2,3,3

        # local = final_x[:,18+30:18+30+2*20*3].reshape(nfrm, 2, 20, 3)

        # # apply R, not R^T
        # positions_rel = np.matmul(local[ :], global_heading_rot[..., None, :, :].transpose(0,1,2,3))
        # # or better with einsum for clarity:
        # # positions_rel = np.einsum('thjc,thck->thjk', local, global_heading_rot)

        # positions_with_heading  = positions_rel.squeeze(-2)
        # velocities_root_xy_no_heading = final_x[:,:2] # frames, 2
        global_heading_rot = final_x[:,6:18].reshape(nfrm, -1, 6) # frames, 6
        global_heading_rot = rotation_6d_to_matrix(torch.from_numpy(global_heading_rot)).numpy()
        # recover global heading
        # global_heading_rot = accumulate_rotations(rotation_6d_to_matrix(torch.from_numpy(global_heading_diff_rot)).numpy())
        inv_global_heading_rot = global_heading_rot # B,2,3,3; B,2,30,3
        # np.transpose(global_heading_rot, (0, 1,3, 2)) 
        # global_heading_rot
        # np.transpose(global_heading_rot, (0, 1,3, 2)) 
        # np.transpose(global_heading_rot, (0, 1,3, 2)) 
        # add global heading to position
        positions_with_heading = np.einsum('binj,bijk->bink',positions_no_heading,np.transpose(global_heading_rot, (0, 1,3, 2)) )
        # positions_with_heading = np.matmul(np.repeat(inv_global_heading_rot[:,:,None], 20, axis=2), positions_no_heading[...,None]).squeeze(-1)
        # print(positions_with_heading.shape) # 60,2,20,3
        # recover root translation
        # add heading to velocities_root_xy_no_heading

        # velocities_root_xyz_no_heading = np.zeros((velocities_root_xy_no_heading.shape[0], 3))
        # velocities_root_xyz_no_heading[:, 0] = velocities_root_xy_no_heading[:, 0]
        # velocities_root_xyz_no_heading[:, 2] = velocities_root_xy_no_heading[:, 1]
        # velocities_root_xyz_no_heading[1:, :] = np.matmul(inv_global_heading_rot[:-1], velocities_root_xyz_no_heading[1:, :,None]).squeeze(-1)
            #   relative_B_pos = position[:,0,0]-position[:,1,0]
            # vel_A = position[1:,1,0]-position[:1,1,0]
            # vel_A = torch.cat([vel_A,torch.zeros_like(vel_A[:1])],0) # pos,A,pos_B Translation,
        vel_A = final_x[:,:3] # right
        vel_B = final_x[:,3:6]
        # root_right = np.concatenate([np.zeros_like(vel_A[:1]),vel_A[:-1]],0)
        root_right = np.cumsum(np.concatenate([np.zeros_like(vel_A[:1]),vel_A[:-1]],axis=0),axis=0)
        root_left = root_right+vel_B
        # print(root_left.shape,root_right.shape)
        
        # root_translation = np.cumsum(velocities_root_xyz_no_heading, axis=0)


        # add root translation
        positions_with_heading[:, 0, :] += root_left.reshape(nfrm,1,3)
        positions_with_heading[:, 1, :] += root_right.reshape(nfrm,1,3)
        roots = np.concatenate([root_left.reshape(nfrm,1,1,3),root_right.reshape(nfrm,1,1,3)],1)
        positions_with_heading = np.concatenate([roots,positions_with_heading],2)
    elif D in [291]:
        # print(fuck)
        nfrm, _ = final_x.shape
        positions_no_heading = final_x[:,18+3+30:18+3+30+2*20*3].reshape(nfrm, 2,-1, 3) # frames, njoints * 3
        # global_heading_rot_6d = final_x[:,6:18].reshape(nfrm, 2, 6)
        # global_heading_rot = rotation_6d_to_matrix(torch.from_numpy(global_heading_rot_6d)).numpy()  # R # N,2,3,3

        # local = final_x[:,18+30:18+30+2*20*3].reshape(nfrm, 2, 20, 3)

        # # apply R, not R^T
        # positions_rel = np.matmul(local[ :], global_heading_rot[..., None, :, :].transpose(0,1,2,3))
        # # or better with einsum for clarity:
        # # positions_rel = np.einsum('thjc,thck->thjk', local, global_heading_rot)

        # positions_with_heading  = positions_rel.squeeze(-2)
        # velocities_root_xy_no_heading = final_x[:,:2] # frames, 2
        global_heading_rot = final_x[:,6+3:18+3].reshape(nfrm, -1, 6) # frames, 6
        global_heading_rot = rotation_6d_to_matrix(torch.from_numpy(global_heading_rot)).numpy()
        # recover global heading
        # global_heading_rot = accumulate_rotations(rotation_6d_to_matrix(torch.from_numpy(global_heading_diff_rot)).numpy())
        inv_global_heading_rot = global_heading_rot # B,2,3,3; B,2,30,3
        # np.transpose(global_heading_rot, (0, 1,3, 2)) 
        # global_heading_rot
        # np.transpose(global_heading_rot, (0, 1,3, 2)) 
        # np.transpose(global_heading_rot, (0, 1,3, 2)) 
        # add global heading to position
        positions_with_heading = np.einsum('binj,bijk->bink',positions_no_heading,np.transpose(global_heading_rot, (0, 1,3, 2)) )
        # positions_with_heading = np.matmul(np.repeat(inv_global_heading_rot[:,:,None], 20, axis=2), positions_no_heading[...,None]).squeeze(-1)
        # print(positions_with_heading.shape) # 60,2,20,3
        # recover root translation
        # add heading to velocities_root_xy_no_heading

        # velocities_root_xyz_no_heading = np.zeros((velocities_root_xy_no_heading.shape[0], 3))
        # velocities_root_xyz_no_heading[:, 0] = velocities_root_xy_no_heading[:, 0]
        # velocities_root_xyz_no_heading[:, 2] = velocities_root_xy_no_heading[:, 1]
        # velocities_root_xyz_no_heading[1:, :] = np.matmul(inv_global_heading_rot[:-1], velocities_root_xyz_no_heading[1:, :,None]).squeeze(-1)
            #   relative_B_pos = position[:,0,0]-position[:,1,0]
            # vel_A = position[1:,1,0]-position[:1,1,0]
            # vel_A = torch.cat([vel_A,torch.zeros_like(vel_A[:1])],0) # pos,A,pos_B Translation,
        vel_A = final_x[:,:3] # right
        vel_B = final_x[:,3:6]
        global_pos = final_x[0:1,6:9]
        # root_right = np.concatenate([np.zeros_like(vel_A[:1]),vel_A[:-1]],0)
        root_right = np.cumsum(np.concatenate([np.zeros_like(vel_A[:1]),vel_A[:-1]],axis=0),axis=0)
        root_left = np.cumsum(np.concatenate([np.zeros_like(vel_B[:1]),vel_B[:-1]],axis=0),axis=0)+global_pos
        # root_left = root_right+vel_B
        # print(root_left.shape,root_right.shape)
        
        # root_translation = np.cumsum(velocities_root_xyz_no_heading, axis=0)


        # add root translation
        positions_with_heading[:, 0, :] += root_left.reshape(nfrm,1,3)
        positions_with_heading[:, 1, :] += root_right.reshape(nfrm,1,3)
        roots = np.concatenate([root_left.reshape(nfrm,1,1,3),root_right.reshape(nfrm,1,1,3)],1)
        positions_with_heading = np.concatenate([roots,positions_with_heading],2)
    elif D in [284,164,134]:
        nfrm, _ = final_x.shape
        positions_no_heading = final_x[:,14:14+2*20*3].reshape(nfrm, 2,-1, 3) # frames, njoints * 3
        # global_heading_rot_6d = final_x[:,6:18].reshape(nfrm, 2, 6)
        # global_heading_rot = rotation_6d_to_matrix(torch.from_numpy(global_heading_rot_6d)).numpy()  # R # N,2,3,3

        # local = final_x[:,18+30:18+30+2*20*3].reshape(nfrm, 2, 20, 3)

        # # apply R, not R^T
        # positions_rel = np.matmul(local[ :], global_heading_rot[..., None, :, :].transpose(0,1,2,3))
        # # or better with einsum for clarity:
        # # positions_rel = np.einsum('thjc,thck->thjk', local, global_heading_rot)

        # positions_with_heading  = positions_rel.squeeze(-2)
        # velocities_root_xy_no_heading = final_x[:,:2] # frames, 2
        global_heading_rot = final_x[:,6:14].reshape(nfrm, -1, 4) # frames, 6
        global_heading_rot = quaternion_to_matrix(torch.from_numpy(global_heading_rot).float()).numpy()
        # recover global heading
        # global_heading_rot = accumulate_rotations(rotation_6d_to_matrix(torch.from_numpy(global_heading_diff_rot)).numpy())
        inv_global_heading_rot = global_heading_rot # B,2,3,3; B,2,30,3
        # np.transpose(global_heading_rot, (0, 1,3, 2)) 
        # global_heading_rot
        # np.transpose(global_heading_rot, (0, 1,3, 2)) 
        # np.transpose(global_heading_rot, (0, 1,3, 2)) 
        # add global heading to position
        positions_with_heading = np.einsum('binj,bijk->bink',positions_no_heading,np.transpose(global_heading_rot, (0, 1,3, 2)) )
        # positions_with_heading = np.matmul(np.repeat(inv_global_heading_rot[:,:,None], 20, axis=2), positions_no_heading[...,None]).squeeze(-1)
        # print(positions_with_heading.shape) # 60,2,20,3
        # recover root translation
        # add heading to velocities_root_xy_no_heading

        # velocities_root_xyz_no_heading = np.zeros((velocities_root_xy_no_heading.shape[0], 3))
        # velocities_root_xyz_no_heading[:, 0] = velocities_root_xy_no_heading[:, 0]
        # velocities_root_xyz_no_heading[:, 2] = velocities_root_xy_no_heading[:, 1]
        # velocities_root_xyz_no_heading[1:, :] = np.matmul(inv_global_heading_rot[:-1], velocities_root_xyz_no_heading[1:, :,None]).squeeze(-1)
            #   relative_B_pos = position[:,0,0]-position[:,1,0]
            # vel_A = position[1:,1,0]-position[:1,1,0]
            # vel_A = torch.cat([vel_A,torch.zeros_like(vel_A[:1])],0) # pos,A,pos_B Translation,
        vel_A = final_x[:,:3] # right
        vel_B = final_x[:,3:6]
        # root_right = np.concatenate([np.zeros_like(vel_A[:1]),vel_A[:-1]],0)
        root_right = np.cumsum(np.concatenate([np.zeros_like(vel_A[:1]),vel_A[:-1]],axis=0),axis=0)
        root_left = root_right+vel_B
        # print(root_left.shape,root_right.shape)
        
        # root_translation = np.cumsum(velocities_root_xyz_no_heading, axis=0)


        # add root translation
        positions_with_heading[:, 0, :] += root_left.reshape(nfrm,1,3)
        positions_with_heading[:, 1, :] += root_right.reshape(nfrm,1,3)
        roots = np.concatenate([root_left.reshape(nfrm,1,1,3),root_right.reshape(nfrm,1,1,3)],1)
        positions_with_heading = np.concatenate([roots,positions_with_heading],2)
    else:
        nfrm, _ = final_x.shape
        positions_no_heading = final_x[:,6+2*16*6:6+2*16*6+2*20*3].reshape(nfrm, 2,-1, 3) # frames, njoints * 3
        # velocities_root_xy_no_heading = final_x[:,:2] # frames, 2
        global_heading_rot = final_x[:,6:6+2*16*6].reshape(nfrm, 2,-1, 6)[:,:,0] # frames, 6
        global_heading_rot = rotation_6d_to_matrix(torch.from_numpy(global_heading_rot)).numpy()
        # recover global heading
        # global_heading_rot = accumulate_rotations(rotation_6d_to_matrix(torch.from_numpy(global_heading_diff_rot)).numpy())
        inv_global_heading_rot = np.transpose(global_heading_rot, (0, 1,3, 2)) 
        # add global heading to position
        positions_with_heading = np.matmul(np.repeat(inv_global_heading_rot[:,:,None], 20, axis=2), positions_no_heading[...,None]).squeeze(-1)
        # print(positions_with_heading.shape) # 60,2,20,3
        # recover root translation
        # add heading to velocities_root_xy_no_heading

        # velocities_root_xyz_no_heading = np.zeros((velocities_root_xy_no_heading.shape[0], 3))
        # velocities_root_xyz_no_heading[:, 0] = velocities_root_xy_no_heading[:, 0]
        # velocities_root_xyz_no_heading[:, 2] = velocities_root_xy_no_heading[:, 1]
        # velocities_root_xyz_no_heading[1:, :] = np.matmul(inv_global_heading_rot[:-1], velocities_root_xyz_no_heading[1:, :,None]).squeeze(-1)
            #   relative_B_pos = position[:,0,0]-position[:,1,0]
            # vel_A = position[1:,1,0]-position[:1,1,0]
            # vel_A = torch.cat([vel_A,torch.zeros_like(vel_A[:1])],0) # pos,A,pos_B Translation,
        vel_A = final_x[:,:3] # right
        vel_B = final_x[:,3:6]
        root_right = np.cumsum(vel_A,axis=0)
        root_left = root_right+vel_B
        # print(root_left.shape,root_right.shape)
        
        # root_translation = np.cumsum(velocities_root_xyz_no_heading, axis=0)


        # add root translation
        positions_with_heading[:, 0, :] += root_left.reshape(nfrm,1,3)
        positions_with_heading[:, 1, :] += root_right.reshape(nfrm,1,3)


    return positions_with_heading.reshape(nfrm,-1,3)



# def recover_from_local_position_torch(final_x, njoint: int):
#     """
#     Recover global joint positions from local motion representation.
    
#     Args:
#         final_x: [B, T, D] motion tensor
#         njoint: number of joints per person (e.g. 20)
        
#     Returns:
#         positions_with_heading: [B, T, num_persons, njoint+1, 3]
#             (with roots added as extra joint 0)
#     """
#     njoint = 20
#     B, T, D = final_x.shape
    
#     # ------------------------------------------------------------
#     # 1. Extract local positions (no heading)
#     # ------------------------------------------------------------
#     # Shape after reshape: [B, T, 2, njoint, 3]
#     positions_no_heading = final_x[:, :, 18:18 + 2 * njoint * 3].reshape(B, T, 2, njoint, 3)
    
#     # ------------------------------------------------------------
#     # 2. Extract 6D rotation and convert to rotation matrices
#     # ------------------------------------------------------------
#     global_heading_6d = final_x[:, :, 6:18].reshape(B, T, -1, 6)  # [B, T, 2, 6]
#     global_heading_rot = rotation_6d_to_matrix(global_heading_6d)  # → [B, T, 2, 3, 3]
    
#     inv_global_heading_rot = global_heading_rot.transpose(-1, -2)  # inverse rotation = transpose
    
#     # ------------------------------------------------------------
#     # 3. Apply heading rotation to local positions
#     # ------------------------------------------------------------
#     # Align shapes for batched matmul: [B, T, 2, njoint, 3, 1]
#     positions_with_heading = torch.matmul(
#         inv_global_heading_rot.unsqueeze(3).repeat(1, 1, 1, njoint, 1, 1),
#         positions_no_heading.unsqueeze(-1)
#     ).squeeze(-1)  # [B, T, 2, njoint, 3]
    
#     # ------------------------------------------------------------
#     # 4. Recover root translation
#     # ------------------------------------------------------------
#     vel_A = final_x[:, :, :3]   # [B, T, 3]
#     vel_B = final_x[:, :, 3:6]  # [B, T, 3]

#     root_right = torch.cumsum(vel_A, dim=1)  # integrate velocity over time
#     root_left  = root_right + vel_B

#     # Add translation to the positions
#     positions_with_heading[:, :, 0, :, :] += root_left.unsqueeze(2)   # person 0 (left)
#     positions_with_heading[:, :, 1, :, :] += root_right.unsqueeze(2)  # person 1 (right)
    
#     # ------------------------------------------------------------
#     # 5. Append roots as an extra "joint" (joint 0)
#     # ------------------------------------------------------------
#     roots = torch.stack([root_left, root_right], dim=2).unsqueeze(3)  # [B, T, 2, 1, 3]
#     positions_with_heading = torch.cat([roots, positions_with_heading], dim=3)  # [B, T, 2, njoint+1, 3]
    
#     return positions_with_heading

def recover_from_local_position_torch(final_x: torch.Tensor, njoint: int):
    """
    Recover global joint positions (two persons) from local representation.

    Args:
        final_x: Tensor of shape [B, T, D], where D is either 258 or 438.
                 - D == 258:
                     [0:6]   -> root velocities (A: 0:3, B: 3:6)
                     [6:18]  -> global heading 6D (2 persons × 6)
                     [18:18+2*njoint*3] -> local positions (no heading), shape (2, njoint, 3)
                 - D == 438:
                     [0:6]   -> root velocities (A: 0:3, B: 3:6)
                     [6:6+2*16*6] -> per-person per-joint 6D rotations (2 persons, 16 joints, 6D)
                     [6+2*16*6 : 6+2*16*6+2*njoint*3] -> local positions (no heading)

        njoint: number of joints per person in positions_no_heading (e.g., 20)

    Returns:
        positions_with_heading: Tensor of shape [B, T, 2, njoint, 3]
            Global positions for two persons (no extra root joint appended).
    """
   
    B, T, D = final_x.shape
    device = final_x.device
    njoint = 20

    # ---- helper: rotation-6d -> rotation-matrix (expects [..., 6] -> [..., 3, 3]) ----
    # If you already have this function, remove this local impl and use yours.
   

    # ---- 1) extract local positions (no heading) ----
    if D == 258:
        # positions_no_heading: [B, T, 2, njoint, 3]
        pos_start = 18
        pos_end = 18 + 2 * njoint * 3
        positions_no_heading = final_x[:, :, pos_start:pos_end].reshape(B, T, 2, njoint, 3)

        # global heading 6D: [B, T, 2, 6]  (2 persons)
        head6d = final_x[:, :, 6:18].reshape(B, T, 2, 6)  # 12 -> (2,6)
        global_heading_rot = rotation_6d_to_matrix(head6d)             # [B, T, 2, 3, 3]

    elif D == 438:
        # positions_no_heading: [B, T, 2, njoint, 3]
        pos_start = 6 + 2 * 16 * 6
        pos_end = pos_start + 2 * njoint * 3
        positions_no_heading = final_x[:, :, pos_start:pos_end].reshape(B, T, 2, njoint, 3)

        # global heading 6D comes from the first joint's 6D per person: [B, T, 2, 6]
        head6d_block = final_x[:, :, 6:6 + 2 * 16 * 6].reshape(B, T, 2, 16, 6)
        head6d = head6d_block[:, :, :, 0, :]                            # pick joint-0 heading
        global_heading_rot = rotation_6d_to_matrix(head6d)              # [B, T, 2, 3, 3]
    else:
        raise ValueError(f"Unsupported D={D}. Expected 258 or 438.")

    # ---- 2) apply inverse heading to local positions ----
    # inv(R) = R^T for rotation matrices
    inv_global_heading_rot = global_heading_rot.transpose(-1, -2)       # [B, T, 2, 3, 3]

    # batched matmul: [B,T,2,1,3,3] x [B,T,2,njoint,3,1] -> [B,T,2,njoint,3,1] -> squeeze
    positions_with_heading = torch.matmul(
        inv_global_heading_rot.unsqueeze(3),                             # [B,T,2,1,3,3]
        positions_no_heading.unsqueeze(-1)                               # [B,T,2,njoint,3,1]
    ).squeeze(-1)                                                        # [B, T, 2, njoint, 3]

    # ---- 3) recover root translations from velocities ----
    # vel_A (right) and vel_B (offset to left): both [B, T, 3]
    vel_A = final_x[:, :, :3]
    vel_B = final_x[:, :, 3:6]
    root_right = torch.cumsum(vel_A, dim=1)      # [B, T, 3]
    root_left  = root_right + vel_B              # [B, T, 3]

    # add translations: broadcast over njoint
    positions_with_heading[:, :, 0, :, :] += root_left.unsqueeze(2)     # person 0
    positions_with_heading[:, :, 1, :, :] += root_right.unsqueeze(2)    # person 1
    roots = torch.cat([root_left.reshape(B,T,1,1,3),root_right.reshape(B,T,1,1,3)],2)
    # print(roots.shape,positions_with_heading.shape)
    positions_with_heading = torch.cat([roots,positions_with_heading],3)
    return positions_with_heading

# add hip height to translation when recoverring from rotation
def recover_from_local_rotation(final_x, njoint):
    nfrm, _ = final_x.shape
    rotations_matrix = rotation_6d_to_matrix(torch.from_numpy(final_x[:,8+6*njoint:8+12*njoint]).reshape(nfrm, -1, 6)).numpy()
    global_heading_diff_rot = final_x[:,2:8]
    velocities_root_xy_no_heading = final_x[:,:2]
    positions_no_heading = final_x[:, 8:8+3*njoint].reshape(nfrm, -1, 3)
    height = positions_no_heading[:, 0, 1]

    global_heading_rot = accumulate_rotations(rotation_6d_to_matrix(torch.from_numpy(global_heading_diff_rot)).numpy())
    inv_global_heading_rot = np.transpose(global_heading_rot, (0, 2, 1))
    # recover root rotation
    rotations_matrix[:,0,...] = np.matmul(inv_global_heading_rot, rotations_matrix[:,0,...])
    velocities_root_xyz_no_heading = np.zeros((velocities_root_xy_no_heading.shape[0], 3))
    velocities_root_xyz_no_heading[:, 0] = velocities_root_xy_no_heading[:, 0]
    velocities_root_xyz_no_heading[:, 2] = velocities_root_xy_no_heading[:, 1]
    velocities_root_xyz_no_heading[1:, :] = np.matmul(inv_global_heading_rot[:-1], velocities_root_xyz_no_heading[1:, :,None]).squeeze(-1)
    root_translation = np.cumsum(velocities_root_xyz_no_heading, axis=0)
    root_translation[:, 1] = height
    smpl_85 = rotations_matrix_to_smpl85(rotations_matrix, root_translation)
    return smpl_85


def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions
    