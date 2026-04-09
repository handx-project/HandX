# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys

from isaacgym import gymapi
import torch


class BaseTask():

    def __init__(self, cfg, enable_camera_sensors=False):

        self.gym = gymapi.acquire_gym()

        self.device_type = cfg.get("device_type", "cuda")
        self.device_id = cfg.get("device_id", 0)

        self.device = "cpu"
        if self.device_type == "cuda" or self.device_type == "GPU":
            self.device = "cuda" + ":" + str(self.device_id)

        self.headless = cfg["headless"]

        self.graphics_device_id = self.device_id
        if self.headless:
            self.graphics_device_id = -1

        self.num_envs = cfg["env"]["numEnvs"]
        self.num_obs = cfg["env"]["numObservations"]
        self.num_actions = cfg["env"]["numActions"]

        self.control_freq_inv = cfg["env"].get("controlFrequencyInv", 1)

        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.start_times = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.data_id = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        self.enable_viewer_sync = True
        self.viewer = None

        if self.headless == False:
            camera_props = gymapi.CameraProperties()
            camera_props.horizontal_fov = 60.0
            camera_props.width = 1600
            camera_props.height = 900
            self.viewer = self.gym.create_viewer(self.sim, camera_props)
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
                cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
            else:
                cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
                cam_target = gymapi.Vec3(10.0, 0.0, 15.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def set_sim_params_up_axis(self, sim_params, axis):
        if axis == 'z':
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity.x = 0
            sim_params.gravity.y = 0
            sim_params.gravity.z = -9.81
            return 2
        return 1

    def create_sim(self, compute_device, graphics_device, physics_engine, sim_params):
        sim = self.gym.create_sim(compute_device, graphics_device, physics_engine, sim_params)
        if sim is None:
            print("*** Failed to create sim")
            quit()
        return sim

    def render(self, sync_frame_time=False, t=0):
        if self.viewer:
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
            else:
                self.gym.poll_viewer_events(self.viewer)
