"""Display poses and trajectories produced by the S17 C++ controller."""
from bisect import bisect_right
import ctypes as C
import os

import mujoco
import numpy as np


class NativePlayback:
    def __init__(self, library, scene, catalogue, robot, clips):
        self.robot = robot
        self.names = [name for name, _ in clips]
        self.ends = np.cumsum([count for _, count in clips]).tolist()
        self.dt = 1./60
        self.cadence = 6
        self.stepper_reanchors = 0
        self.velocity = np.zeros(robot.model.nv)
        self.library = C.CDLL(str(library))
        double_pointer = C.POINTER(C.c_double)
        signatures = {
            "s17_search_create": (C.c_void_p, C.c_char_p, C.c_char_p, C.POINTER(C.c_int),
                                  C.POINTER(C.c_char), C.c_size_t),
            "s17_search_destroy": (None, C.c_void_p),
            "s17_search_gamepad": (None, C.c_void_p, C.POINTER(C.c_float), C.c_int, C.c_int, C.c_int, C.c_float),
            "s17_search_velocity": (None, C.c_void_p, C.c_float, C.c_float, C.c_float, C.c_int),
            "s17_reference_step": (None, C.c_void_p, C.c_float, C.c_float, C.c_float),
            "s17_reference_forecast": (None, C.c_void_p, C.c_float, C.c_float, C.c_float,
                                       C.c_int, double_pointer),
            "s17_search_snapshot": (None, C.c_void_p, C.POINTER(C.c_int), double_pointer,
                                    double_pointer, double_pointer, double_pointer),
        }
        for name, (result, *arguments) in signatures.items():
            function = getattr(self.library, name)
            function.restype, function.argtypes = result, arguments
        frames = C.c_int()
        error = C.create_string_buffer(1024)
        self.handle = self.library.s17_search_create(os.fsencode(scene), os.fsencode(catalogue),
                                                     C.byref(frames), error, len(error))
        if not self.handle:
            raise RuntimeError(error.value.decode(errors="replace") or "C++ matcher initialization failed")
        expected_frames = self.ends[-1] if self.ends else 0
        if frames.value != expected_frames:
            self.close()
            raise ValueError(f"C++ matcher loaded {frames.value} frames, expected {expected_frames}")
        self.result = (C.c_int*2)()
        self.pose_buffer = (C.c_double*robot.model.nq)()
        self.origin_buffer = (C.c_double*2)()
        self.path_buffer = (C.c_double*24)()
        self.command_buffer = (C.c_double*4)()
        self.pose = np.ctypeslib.as_array(self.pose_buffer)
        self.trajectory_origin = np.ctypeslib.as_array(self.origin_buffer)
        self.paths = np.ctypeslib.as_array(self.path_buffer).reshape(2, 3, 4)
        self.command = np.ctypeslib.as_array(self.command_buffer)[:3]
        self._read_state()
        robot.data.qpos[:] = self.pose
        mujoco.mj_forward(robot.model, robot.data)

    @property
    def frame(self):
        return self.result[0]

    @property
    def switches(self):
        return self.result[1]

    @property
    def clip_name(self):
        return self.names[bisect_right(self.ends, self.frame)]

    @property
    def target_heading(self):
        return self.command_buffer[3]

    def _read_state(self):
        self.library.s17_search_snapshot(self.handle, self.result, self.pose_buffer,
                                         self.origin_buffer, self.path_buffer, self.command_buffer)

    def step(self, command, *, gamepad_input=None, hold_heading=False):
        previous = self.pose.copy()
        if gamepad_input is None:
            self.library.s17_search_velocity(self.handle, *command, hold_heading)
        else:
            axes, walk, orbit, camera_yaw = gamepad_input
            self.library.s17_search_gamepad(self.handle, (C.c_float*4)(*axes), walk, orbit, hold_heading, camera_yaw)
        self._read_state()
        mujoco.mj_differentiatePos(self.robot.model, self.velocity, self.dt, previous, self.pose)
        data = self.robot.data
        data.qpos[:] = self.pose
        data.qvel[:] = self.velocity
        data.time += self.dt
        mujoco.mj_forward(self.robot.model, data)

    def trajectories(self, command, heading_target):
        return [(path[:, :2], path[:, 2:]) for path in self.paths]

    def advance_reference(self, command, steps):
        for _ in range(steps):
            self.library.s17_reference_step(self.handle, *command)

    def forecast_reference(self, command, steps):
        poses = np.empty((steps+1, self.robot.model.nq), dtype=np.float64)
        self.library.s17_reference_forecast(self.handle, *command, steps,
                                            poses.ctypes.data_as(C.POINTER(C.c_double)))
        return poses

    def close(self):
        if self.handle:
            self.library.s17_search_destroy(self.handle)
            self.handle = None
