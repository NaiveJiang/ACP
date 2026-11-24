"""Show a simulated full-body rollout next to its reference in MuJoCo."""

import threading
import time

import glfw
import mujoco
import mujoco.viewer
import numpy as np

from ..rotations import mat_to_quat
from .motion import CONTROL_DT
from .render import scene_xml


class ReferenceViewer:
    def __init__(self, motion, boxes, *, primitive_soles=False):
        self.model = mujoco.MjModel.from_xml_string(scene_xml(boxes, primitive_soles=primitive_soles))
        self.data = mujoco.MjData(self.model)
        self.motion = motion
        self.paused = False
        self.last_frame_time = None

    def key_callback(self, key):
        if key == glfw.KEY_SPACE:
            self.paused = not self.paused

    def __enter__(self):
        before_launch = set(threading.enumerate())
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback)
        self.render_threads = set(threading.enumerate()) - before_launch
        with self.viewer.lock():
            self.viewer.cam.distance = 3.2
            self.viewer.cam.elevation = -15.
            self.viewer.cam.azimuth = 135.
        return self

    def __exit__(self, *exc):
        self.viewer.close()
        # Wait for the passive viewer to release GLFW before interpreter shutdown.
        for thread in self.render_threads:
            thread.join(timeout=5.)

    def show(self, state, frame, *, follow_reference=False):
        p, r, _, _, j, _ = self.motion.ref_state(frame)
        ref = np.concatenate((p[0].cpu().numpy(), mat_to_quat(r)[0].cpu().numpy(), j[0].cpu().numpy()))
        clip = self.motion.clip_id[frame]
        time_s = int((frame-self.motion.clip_start[clip])[0]) * CONTROL_DT
        return self.show_reference(state, ref, time_s, follow_reference=follow_reference)

    def show_reference(self, state, reference, time_s, *, follow_reference=False):
        if self.last_frame_time is not None:
            time.sleep(max(0., self.last_frame_time + CONTROL_DT - time.monotonic()))
        if not self.viewer.is_running():
            return False
        position, rotation, _, _, joints, _ = state
        q = np.concatenate((position[0].cpu().numpy(), mat_to_quat(rotation)[0].cpu().numpy(),
                            joints[0].cpu().numpy()))
        with self.viewer.lock():
            self.data.qpos[:] = np.concatenate((q, reference))
            self.data.time = time_s
            camera_target = reference if follow_reference else q
            self.viewer.cam.lookat[:] = (*camera_target[:2], .6)
            mujoco.mj_kinematics(self.model, self.data)
        self.viewer.sync()
        while self.paused and self.viewer.is_running():
            self.viewer.sync()
            time.sleep(.02)
        self.last_frame_time = time.monotonic()
        return self.viewer.is_running()

    def wait_until_closed(self):
        while self.viewer.is_running():
            self.viewer.sync()
            time.sleep(.02)
