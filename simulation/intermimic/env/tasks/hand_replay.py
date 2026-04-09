# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import numpy as np
import torch
import os, random

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
import torch.nn.functional as F
from env.tasks.base_task import BaseTask

class BihandSim(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine

        self._pd_control = self.cfg["env"]["pdControl"]
        self.power_scale = self.cfg["env"]["powerScale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        key_bodies = self.cfg["env"]["keyBodies"]
        self._setup_character_props(key_bodies)

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
         
        super().__init__(cfg=self.cfg)
        
        self.dt = self.control_freq_inv * sim_params.dt
        
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        num_actors = self.get_num_actors_per_env()
        
        # ====================================================================
        # TWO-HAND CHANGE: Two humanoid actors (left hand=0, right hand=1)
        # ====================================================================
        self._all_root_states = self._root_states.view(self.num_envs, num_actors, actor_root_state.shape[-1])
        
        # Left hand is actor 0, right hand is actor 1
        self._left_hand_root_states = self._all_root_states[..., 0, :]
        self._right_hand_root_states = self._all_root_states[..., 1, :]
        
        # For backward compat, _humanoid_root_states points to left hand
        self._humanoid_root_states = self._left_hand_root_states
        
        self._initial_left_hand_root_states = self._left_hand_root_states.clone()
        self._initial_left_hand_root_states[:, 7:13] = 0
        self._initial_right_hand_root_states = self._right_hand_root_states.clone()
        self._initial_right_hand_root_states[:, 7:13] = 0
        self._initial_humanoid_root_states = self._initial_left_hand_root_states

        # Actor IDs for both hands
        self._left_hand_actor_ids = num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        self._right_hand_actor_ids = num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32) + 1
        self._humanoid_actor_ids = self._left_hand_actor_ids  # backward compat
        
        # Both hand actor IDs combined (for setting both at once)
        self._both_hand_actor_ids = torch.stack([self._left_hand_actor_ids, self._right_hand_actor_ids], dim=-1).reshape(-1)

        # create some wrapper tensors for different slices
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        
        # Total DOFs = left_hand_dof + right_hand_dof
        # Left hand DOFs: 0..num_dof_left-1, Right hand DOFs: num_dof_left..num_dof-1
        self._dof_pos = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 0]
        self._dof_vel = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 1]
        
        # Split DOF into left and right
        self._left_dof_pos = self._dof_pos[..., :self.num_dof_left]
        self._left_dof_vel = self._dof_vel[..., :self.num_dof_left]
        self._right_dof_pos = self._dof_pos[..., self.num_dof_left:]
        self._right_dof_vel = self._dof_vel[..., self.num_dof_left:]
        
        self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)
        self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)
        
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)

        # Total bodies = left_hand_bodies + right_hand_bodies
        self._rigid_body_pos = rigid_body_state_reshaped[..., :self.num_bodies, 0:3]
        self._rigid_body_rot = rigid_body_state_reshaped[..., :self.num_bodies, 3:7]
        self._rigid_body_vel = rigid_body_state_reshaped[..., :self.num_bodies, 7:10]
        self._rigid_body_ang_vel = rigid_body_state_reshaped[..., :self.num_bodies, 10:13]

        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., :self.num_bodies, :]
        
        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        
        self._build_termination_heights()
        
        contact_bodies = self.cfg["env"]["contactBodies"]
        self._key_body_ids = self._build_key_body_ids_tensor(key_bodies)
        self._contact_body_ids = self._build_contact_body_ids_tensor(contact_bodies)
        
        if self.viewer != None:
            self._init_camera()
            
        return

    def get_obs_size(self):
        return self._num_obs

    def get_action_size(self):
        return self._num_actions

    def get_num_actors_per_env(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        return num_actors

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        return

    def reset(self, env_ids=None):
        if (env_ids is None):
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        self._reset_envs(env_ids)
        return

    def _reset_envs(self, env_ids):
        if (len(env_ids) > 0):
            self._reset_actors(env_ids)
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
        return

    def _reset_env_tensors(self, env_ids):
        # Set root states and DOF states for BOTH hand actors
        # We need to set both left and right hand actor ids
        left_ids = self._left_hand_actor_ids[env_ids]
        right_ids = self._right_hand_actor_ids[env_ids]
        both_ids = torch.cat([left_ids, right_ids])
        
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(both_ids), len(both_ids))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(both_ids), len(both_ids))
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def _setup_character_props(self, key_bodies):
        asset_file = self.robot_type
        num_key_bodies = len(key_bodies)

        # Each hand has 16 joints * 3 + 1 wrist * 3 = 51 DOFs (hinge)
        # Plus 3 slide DOFs for wrist position = 54 DOFs per hand
        # Total actions: left 51 hinge + right 51 hinge = still 34*3 for the policy
        # (slide joints are not actuated by policy, only set directly)
        self._dof_obs_size = (34)*3
        self._num_actions = (34)*3
        self._num_obs = self.cfg["env"]["numObs"]

        return

    def _build_termination_heights(self):
        self._termination_heights = 0.3
        self._termination_heights = to_torch(self._termination_heights, device=self.device)
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        
        # ====================================================================
        # TWO-HAND CHANGE: Load two separate assets
        # ====================================================================
        left_hand_file = self.robot_type_left
        right_hand_file = self.robot_type_right
        
        left_path = os.path.join(asset_root, left_hand_file)
        right_path = os.path.join(asset_root, right_hand_file)
        
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        left_hand_asset = self.gym.load_asset(self.sim, os.path.dirname(left_path), os.path.basename(left_path), asset_options)
        right_hand_asset = self.gym.load_asset(self.sim, os.path.dirname(right_path), os.path.basename(right_path), asset_options)

        self.num_left_bodies = self.gym.get_asset_rigid_body_count(left_hand_asset)
        self.num_right_bodies = self.gym.get_asset_rigid_body_count(right_hand_asset)
        self.num_humanoid_bodies = self.num_left_bodies + self.num_right_bodies
        
        self.num_left_shapes = self.gym.get_asset_rigid_shape_count(left_hand_asset)
        self.num_right_shapes = self.gym.get_asset_rigid_shape_count(right_hand_asset)
        self.num_humanoid_shapes = self.num_left_shapes + self.num_right_shapes
        
        self.num_dof_left = self.gym.get_asset_dof_count(left_hand_asset)
        self.num_dof_right = self.gym.get_asset_dof_count(right_hand_asset)
        
        # Collect motor efforts from both
        left_actuator_props = self.gym.get_asset_actuator_properties(left_hand_asset)
        right_actuator_props = self.gym.get_asset_actuator_properties(right_hand_asset)
        motor_efforts = [prop.motor_effort for prop in left_actuator_props] + [prop.motor_effort for prop in right_actuator_props]

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        self.num_bodies = self.num_left_bodies + self.num_right_bodies
        self.num_dof = self.num_dof_left + self.num_dof_right
        self.num_joints = self.gym.get_asset_joint_count(left_hand_asset) + self.gym.get_asset_joint_count(right_hand_asset)

        self.left_hand_handles = []
        self.right_hand_handles = []
        self.humanoid_handles = []  # backward compat: stores left hand handles
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        max_agg_bodies = self.num_humanoid_bodies + 2
        max_agg_shapes = self.num_humanoid_shapes + 2        
        
        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            self._build_env(i, env_ptr, left_hand_asset, right_hand_asset)

            self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)

        # Collect DOF limits from both hands
        left_dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.left_hand_handles[0])
        right_dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.right_hand_handles[0])
        
        for dof_prop in [left_dof_prop, right_dof_prop]:
            for j in range(len(dof_prop['lower'])):
                if dof_prop['lower'][j] > dof_prop['upper'][j]:
                    self.dof_limits_lower.append(dof_prop['upper'][j])
                    self.dof_limits_upper.append(dof_prop['lower'][j])
                else:
                    self.dof_limits_lower.append(dof_prop['lower'][j])
                    self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        if (self._pd_control):
            self._build_pd_action_offset_scale()

        return
    
    def _build_env(self, env_id, env_ptr, left_hand_asset, right_hand_asset):
        col_group = env_id
        col_filter = self._get_humanoid_collision_filter()
        segmentation_id = 0

        start_pose = gymapi.Transform()
        char_h = 0.89
        start_pose.p = gymapi.Vec3(*get_axis_params(char_h, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # ====================================================================
        # TWO-HAND CHANGE: Create two actors - left hand and right hand
        # ====================================================================
        left_handle = self.gym.create_actor(env_ptr, left_hand_asset, start_pose, "left_hand", col_group, 1, segmentation_id)
        right_handle = self.gym.create_actor(env_ptr, right_hand_asset, start_pose, "right_hand", col_group, 1, segmentation_id)

        self.gym.enable_actor_dof_force_sensors(env_ptr, left_handle)
        self.gym.enable_actor_dof_force_sensors(env_ptr, right_handle)
        
        print(f"Env {env_id}: left_bodies={self.num_left_bodies}, right_bodies={self.num_right_bodies}")
        
        for j in range(self.num_left_bodies):
            self.gym.set_rigid_body_color(env_ptr, left_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.54, 0.85, 0.2))
        for j in range(self.num_right_bodies):
            self.gym.set_rigid_body_color(env_ptr, right_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.2, 0.54, 0.85))

        if (self._pd_control):
            left_dof_prop = self.gym.get_asset_dof_properties(left_hand_asset)
            left_dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            self.gym.set_actor_dof_properties(env_ptr, left_handle, left_dof_prop)
            
            right_dof_prop = self.gym.get_asset_dof_properties(right_hand_asset)
            right_dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            self.gym.set_actor_dof_properties(env_ptr, right_handle, right_dof_prop)

        self.left_hand_handles.append(left_handle)
        self.right_hand_handles.append(right_handle)
        self.humanoid_handles.append(left_handle)  # backward compat

        return

    def _build_pd_action_offset_scale(self):
        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)
        return

    def _get_humanoid_collision_filter(self):
        return 0

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        return

    def _reset_actors(self, env_ids):
        self._left_hand_root_states[env_ids] = self._initial_left_hand_root_states[env_ids]
        self._right_hand_root_states[env_ids] = self._initial_right_hand_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids] 
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        return

    def render(self, sync_frame_time=False, t=0):
        if self.viewer:
            if not (hasattr(self, 'grid_play_mode') and self.grid_play_mode):
                self._update_camera()
        super().render(sync_frame_time, t=t)
        return

    def _build_key_body_ids_tensor(self, key_body_names):
        env_ptr = self.envs[0]
        body_ids = []

        for body_name in key_body_names:
            # Try left hand first, then right hand
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, self.left_hand_handles[0], body_name)
            if body_id == -1:
                body_id = self.gym.find_actor_rigid_body_handle(env_ptr, self.right_hand_handles[0], body_name)
                if body_id != -1:
                    # Offset by left hand body count since right hand bodies come after
                    body_id = body_id  # rigid body indices are global per env
            
            print(body_id, body_name)
            if body_id < 23:
                body_id = body_id - 4
            else:
                body_id = body_id - 7
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_contact_body_ids_tensor(self, contact_body_names):
        env_ptr = self.envs[0]
        body_ids = []

        for body_name in contact_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, self.left_hand_handles[0], body_name)
            if body_id == -1:
                body_id = self.gym.find_actor_rigid_body_handle(env_ptr, self.right_hand_handles[0], body_name)
            
            if body_id < 23:
                body_id = body_id - 4
            else:
                body_id = body_id - 7
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._left_hand_root_states[0, 0:3].cpu().numpy()

        if hasattr(self, 'grid_play_mode') and self.grid_play_mode:
            # Wide camera to see the full grid
            num_per_row = int(np.ceil(np.sqrt(self.num_envs)))
            num_rows = int(np.ceil(self.num_envs / num_per_row))
            s = self.grid_spacing
            grid_w = (num_per_row - 1) * 2 * s
            grid_d = (num_rows - 1) * 2 * s
            cx = grid_w / 2.0
            cy = grid_d / 2.0
            extent = max(grid_w, grid_d, 1.0)
            cam_dist = extent * 0.9 + 4.0
            cam_z = extent * 0.6 + 2.0
            cam_pos = gymapi.Vec3(cx + 1.0, cy - cam_dist, cam_z - 0.5)
            cam_target = gymapi.Vec3(cx + 1.0, cy, 1.0)
        else:
            cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0],
                                  self._cam_prev_char_pos[1] - 1.0,
                                  1.5)
            cam_target = gymapi.Vec3(self._cam_prev_char_pos[0],
                                     self._cam_prev_char_pos[1],
                                     0.9)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # Override existing lights for grid mode: brighter, softer shadows
        if hasattr(self, 'grid_play_mode') and self.grid_play_mode:
            self.gym.set_light_parameters(
                self.sim, 0,
                gymapi.Vec3(1.0, 1.0, 1.0),   # intensity
                gymapi.Vec3(0.5, 0.5, 0.5),   # medium ambient -> softer shadows
                gymapi.Vec3(0.0, 0.0, 1.0))   # straight from above
            self.gym.set_light_parameters(
                self.sim, 1,
                gymapi.Vec3(0.0, 0.0, 0.0),   # disabled
                gymapi.Vec3(0.0, 0.0, 0.0),
                gymapi.Vec3(0.0, 0.0, 1.0))
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._left_hand_root_states[0, 0:3].cpu().numpy()

        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0],
                                  char_root_pos[1] + cam_delta[1],
                                  cam_pos[2])

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)
        self._cam_prev_char_pos[:] = char_root_pos
        return
    
class HandReplay(BihandSim):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.motion_file = cfg['env']['motion_file']
        self.save_images = cfg['env']['saveImages']

        if os.path.isfile(self.motion_file):
            self.motion_file = [self.motion_file]
        else:
            motion_file = sorted(os.listdir(self.motion_file))
            self.motion_file = sorted([os.path.join(self.motion_file, f) for f in motion_file if f.endswith('.pt')])

        # Grid play: randomly sample grid_n * grid_m sequences
        self.grid_n = cfg['env'].get('gridN', 0)
        self.grid_m = cfg['env'].get('gridM', 0)
        self.grid_spacing = cfg['env'].get('gridSpacing', 1.5)
        self.grid_play_mode = (self.grid_n > 0 and self.grid_m > 0)
        if self.grid_play_mode:
            n_total = self.grid_n * self.grid_m
            grid_seed = cfg['env'].get('gridSeed', -1)
            rng = random.Random(grid_seed if grid_seed >= 0 else None)
            if len(self.motion_file) >= n_total:
                self.motion_file = rng.sample(self.motion_file, n_total)
            else:
                self.motion_file = rng.choices(self.motion_file, k=n_total)
            self.motion_file = sorted(self.motion_file)
            cfg['env']['envSpacing'] = self.grid_spacing
            print(f"[Grid Play] {self.grid_n}x{self.grid_m} grid, spacing={self.grid_spacing}m, {n_total} sequences")

        self.robot_type = cfg['env']['robotType']
        self.robot_type_left = cfg['env'].get('robotTypeLeft', self.robot_type.replace('_hand.xml', '_lhand.xml'))
        self.robot_type_right = cfg['env'].get('robotTypeRight', self.robot_type.replace('_hand.xml', '_rhand.xml'))

        print(f"Left hand XML:  {self.robot_type_left}")
        print(f"Right hand XML: {self.robot_type_right}")

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        # Env world origins for grid play (offset hand positions per env)
        num_per_row = int(np.ceil(np.sqrt(self.num_envs)))
        s = cfg['env'].get('envSpacing', 5.0)
        env_origins = []
        for i in range(self.num_envs):
            col = i % num_per_row
            row = i // num_per_row
            env_origins.append([col * 2.0 * s, row * 2.0 * s, 0.0])
        self._env_origins = torch.tensor(env_origins, device=self.device, dtype=torch.float)

        self._load_motion(self.motion_file)
        return

    def _load_motion(self, motion_file):
        """Load .pt files directly — only wrist pos + DOFs (first 102 cols) are used."""
        if not isinstance(motion_file, list):
            motion_file = [motion_file]
        self.num_motions = len(motion_file)
        self.hoi_data_dict = []
        self.max_episode_length = []
        hoi_datas = []
        for data_path in motion_file:
            data = torch.load(data_path, weights_only=False).detach().to(self.device)
            self.max_episode_length.append(data.shape[0])
            self.hoi_data_dict.append({'hoi_data': data})
            hoi_datas.append(data)
        max_length = max(self.max_episode_length)
        self.max_episode_length = to_torch(self.max_episode_length, dtype=torch.long)
        padded = [F.pad(d, (0, 0, 0, max_length - d.shape[0])) for d in hoi_datas]
        self.hoi_data = torch.stack(padded, dim=0)  # [num_motions, max_T, 245]
        return

    def play_dataset_step(self, time):
        t = time
        env_ids = to_torch([i for i in range(self.num_envs)], device=self.device, dtype=torch.long)

        self._dof_vel[:] = torch.zeros_like(self._dof_vel[:])

        if self.grid_play_mode:
            # ----- grid mode: each env plays its own sequence -----
            t_per_env = torch.tensor(
                [time % self.max_episode_length[i].item() for i in range(self.num_envs)],
                device=self.device, dtype=torch.long)
            env_idx = torch.arange(self.num_envs, device=self.device)
            # hoi_data: [num_motions, max_T, obs_size]; first 102 cols = wrist+dof data
            ref_data = self.hoi_data[env_idx, t_per_env]          # [num_envs, obs_size]
            z_adjust = torch.tensor([0.0, 0.0, -0.7], device=self.device)
            left_wrist_pos  = ref_data[:, :3]    + self._env_origins + z_adjust
            right_wrist_pos = ref_data[:, 51:54] + self._env_origins + z_adjust
            left_dofs  = torch.cat([torch.zeros(self.num_envs, 3, device=self.device), ref_data[:, 3:51]],  dim=1)
            right_dofs = torch.cat([torch.zeros(self.num_envs, 3, device=self.device), ref_data[:, 54:102]], dim=1)
            self._dof_pos[:] = torch.cat([left_dofs, right_dofs], dim=1)
        else:
            # ----- single-sequence mode -----
            ref_dof = self.hoi_data_dict[0]['hoi_data'][t, :102].clone()
            left_wrist_pos  = ref_dof[:3].unsqueeze(0).expand(self.num_envs, -1)
            right_wrist_pos = ref_dof[51:54].unsqueeze(0).expand(self.num_envs, -1)
            left_dof  = torch.cat([torch.zeros(3, device=self.device), ref_dof[3:51]])
            right_dof = torch.cat([torch.zeros(3, device=self.device), ref_dof[54:102]])
            self._dof_pos[:] = torch.cat([left_dof, right_dof]).unsqueeze(0).expand(self.num_envs, -1)

        self._left_hand_root_states[:, 0:3]  = left_wrist_pos
        self._left_hand_root_states[:, 3:6]  = 0.0
        self._left_hand_root_states[:, 6]    = 1.0
        self._left_hand_root_states[:, 7:13] = 0.0
        self._right_hand_root_states[:, 0:3]  = right_wrist_pos
        self._right_hand_root_states[:, 3:6]  = 0.0
        self._right_hand_root_states[:, 6]    = 1.0
        self._right_hand_root_states[:, 7:13] = 0.0
        
        # Set both hand actor states
        left_ids = self._left_hand_actor_ids[env_ids]
        right_ids = self._right_hand_actor_ids[env_ids]
        both_ids = torch.cat([left_ids, right_ids])
        
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(both_ids), len(both_ids))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(both_ids), len(both_ids))

        self._refresh_sim_tensors()

        # Lock camera for grid mode at every frame (overrides _update_camera)
        if self.grid_play_mode and self.viewer:
            num_per_row = int(np.ceil(np.sqrt(self.num_envs)))
            num_rows = int(np.ceil(self.num_envs / num_per_row))
            s = self.grid_spacing
            grid_w = (num_per_row - 1) * 2 * s
            grid_d = (num_rows - 1) * 2 * s
            cx = grid_w / 2.0 + 2.0
            cy = grid_d / 2.0
            extent = max(grid_w, grid_d, 1.0)
            cam_dist = extent * 0.9 + 4.0 - 4.0
            cam_z = extent * 0.6 + 1.5 - 0.0
            self.gym.viewer_camera_look_at(
                self.viewer, None,
                gymapi.Vec3(cx, cy - cam_dist, cam_z),
                gymapi.Vec3(cx, cy, -1.0))

        self.render(t=t)
        self.gym.simulate(self.sim)
        return

    def render(self, sync_frame_time=False, t=0):
        super().render(sync_frame_time, t=t)
        if self.viewer:
            self._draw_task()
            if self.save_images:
                env_ids = 0
                frame_id = t
                dataname = 'example'
                rgb_filename = "physhoi/data/images/" + dataname + "/rgb_env%d_frame%05d.png" % (env_ids, frame_id)
                os.makedirs("physhoi/data/images/" + dataname, exist_ok=True)
                self.gym.write_viewer_image_to_file(self.viewer, rgb_filename)
        return
    
    def _draw_task(self):
        return