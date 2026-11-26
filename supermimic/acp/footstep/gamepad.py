"""SDL gamepad input and stop/start handling for the simulator."""
import ctypes as C
from ctypes.util import find_library
import platform
from pathlib import Path
import time

import torch

from ...rotations import yaw_mat
from ..motion import CONTROL_DT


class Gamepad:
    _SUBSYSTEM = 0x2000  # SDL_INIT_GAMEPAD
    _LEFT_SHOULDER = 9
    _RIGHT_SHOULDER = 10  # SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER
    _START = 6
    _DPAD_DOWN = 12
    _DPAD_LEFT = 13
    _DPAD_RIGHT = 14
    _LEFT_TRIGGER = 4
    _RIGHT_TRIGGER = 5

    def __init__(self, library=None, *, wait=0.):
        vendored = (Path.home() / "lejulab_platform/src/leju-joystick/3rd_party"
                    / platform.machine() / "SDL/lib/libSDL3.so.0")
        library = library or find_library("SDL3") or (str(vendored) if vendored.is_file() else None)
        if not library:
            raise RuntimeError("SDL3 not found; supply --gamepad-library PATH")
        self.sdl = C.CDLL(library)
        signatures = {
            "SDL_InitSubSystem": (C.c_bool, C.c_uint32),
            "SDL_QuitSubSystem": (None, C.c_uint32),
            "SDL_GetError": (C.c_char_p,),
            "SDL_GetGamepads": (C.POINTER(C.c_uint32), C.POINTER(C.c_int)),
            "SDL_free": (None, C.c_void_p),
            "SDL_OpenGamepad": (C.c_void_p, C.c_uint32),
            "SDL_CloseGamepad": (None, C.c_void_p),
            "SDL_GetGamepadName": (C.c_char_p, C.c_void_p),
            "SDL_GamepadConnected": (C.c_bool, C.c_void_p),
            "SDL_GetGamepadAxis": (C.c_int16, C.c_void_p, C.c_int),
            "SDL_GetGamepadButton": (C.c_bool, C.c_void_p, C.c_int),
            "SDL_UpdateGamepads": (None,),
        }
        for name, (result, *arguments) in signatures.items():
            function = getattr(self.sdl, name)
            function.restype, function.argtypes = result, arguments
        if not self.sdl.SDL_InitSubSystem(self._SUBSYSTEM):
            raise RuntimeError(self.sdl.SDL_GetError().decode())
        self.handle = None
        self.a_pressed = True  # A must be released before it can reset.
        self.start_pressed = True
        try:
            deadline = time.monotonic() + wait
            while not self.handle:
                self.sdl.SDL_UpdateGamepads()
                count = C.c_int()
                devices = self.sdl.SDL_GetGamepads(C.byref(count))
                try:
                    for index in range(count.value):
                        self.handle = self.sdl.SDL_OpenGamepad(devices[index])
                        if self.handle:
                            break
                finally:
                    self.sdl.SDL_free(devices)
                if self.handle:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("No SDL gamepad detected; connect or power on the controller and retry")
                time.sleep(min(.1, remaining))
            name = self.sdl.SDL_GetGamepadName(self.handle)
            self.name = name.decode(errors="replace").strip() if name else "SDL gamepad"
        except Exception:
            self.close()
            raise

    def read(self, *, full_right=False):
        self.sdl.SDL_UpdateGamepads()
        if not self.sdl.SDL_GamepadConnected(self.handle):
            raise RuntimeError("Gamepad disconnected; reopen playback after reconnecting")
        # Return left forward/left, right left, and optionally right forward.
        sdl_axes = (1, 0, 2, 3) if full_right else (1, 0, 2)
        axes = tuple(min(1., -self.sdl.SDL_GetGamepadAxis(self.handle, axis) / 32767.)
                     for axis in sdl_axes)
        pressed = self.sdl.SDL_GetGamepadButton(self.handle, 0)
        reset = pressed and not self.a_pressed
        self.a_pressed = pressed
        return axes, reset

    def run_pressed(self):
        return self.sdl.SDL_GetGamepadButton(self.handle, self._RIGHT_SHOULDER)

    def start_edge(self):
        pressed = self.sdl.SDL_GetGamepadButton(self.handle, self._START)
        edge = pressed and not self.start_pressed
        self.start_pressed = pressed
        return edge

    def motion_matching_buttons(self):
        walk = self.a_pressed  # South button, sampled by read().
        orbit = self.sdl.SDL_GetGamepadAxis(self.handle, self._RIGHT_TRIGGER) > .425*32767
        left = self.sdl.SDL_GetGamepadButton(self.handle, self._LEFT_SHOULDER)
        right = self.sdl.SDL_GetGamepadButton(self.handle, self._RIGHT_SHOULDER)
        return walk, orbit, int(left)-int(right)

    def hold_heading_pressed(self):
        return self.sdl.SDL_GetGamepadAxis(self.handle, self._LEFT_TRIGGER) > .425*32767

    def turning_inputs(self):
        dpad = tuple(self.sdl.SDL_GetGamepadButton(self.handle, button)
                     for button in (self._DPAD_LEFT, self._DPAD_RIGHT, self._DPAD_DOWN))
        triggers = tuple(max(0, self.sdl.SDL_GetGamepadAxis(self.handle, axis)) / 32767.
                         for axis in (self._LEFT_TRIGGER, self._RIGHT_TRIGGER))
        return dpad, triggers

    def close(self):
        if self.handle:
            self.sdl.SDL_CloseGamepad(self.handle)
            self.handle = None
        self.sdl.SDL_QuitSubSystem(self._SUBSYSTEM)


class StickCommand:
    """Map the dominant stick axis into the clock's signed command range."""

    def __init__(self, commands):
        caps = ((.5, .4), (.3, .3), (.7, .7))
        self.ranges = {}
        for axis, limits in enumerate(caps):
            others = [i for i in range(3) if i != axis]
            pure = commands[(commands[:, others].abs() < 1e-5).all(dim=1), axis]
            for sign, cap in zip((1, -1), limits):
                magnitudes = (pure * sign)[pure * sign > 1e-5]
                if len(magnitudes) and float(magnitudes.min()) <= cap:
                    self.ranges[axis, sign] = float(magnitudes.min()), min(cap, float(magnitudes.max()))
        self.reset()

    def reset(self):
        self.centered = False

    def update(self, axes):
        deadzone = .18
        axis = max(range(3), key=lambda i: abs(axes[i]))
        if abs(axes[axis]) <= deadzone:
            self.centered = True
            return None
        if not self.centered:
            return None
        direction = axis, 1 if axes[axis] > 0 else -1
        if direction not in self.ranges:
            return None
        lo, hi = self.ranges[direction]
        fraction = round(20 * (abs(axes[axis]) - deadzone) / (1 - deadzone)) / 20
        command = [0., 0., 0.]
        command[axis] = direction[1] * (lo + fraction * (hi - lo))
        return direction, command


class GamepadGait:
    """Brake with the learned gait, then hold both feet in world coordinates."""

    def __init__(self, planner, ground, stick):
        self.planner = planner
        self.ground = ground
        self.stick = stick

    def _capture_stance(self, state):
        feet, rotation, gap, _ = self.ground.foot_state(state)
        self.standing_feet = feet.clone()
        self.standing_feet[..., 2] -= gap.clamp_min(0.)
        self.standing_yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])

    def reset(self, state):
        self.stick.reset()
        self.direction = None
        self.braking = self.settling = 0
        self._capture_stance(state)

    def _standing_targets(self, state):
        basis = yaw_mat(state[1])
        local = (basis.transpose(-1, -2)[:, None] @
                 (self.standing_feet-state[0][:, None])[..., None]).squeeze(-1)
        angle = self.standing_yaw-torch.atan2(basis[:, 1, 0], basis[:, 0, 0])[:, None]
        return local, torch.stack((angle.sin(), angle.cos()), dim=-1)

    def step(self, observation, state, request, query_dt, body_yaw):
        if self.direction is not None:
            if request is not None and request[0] == self.direction:
                self.braking = 0
                return self.planner.step(observation, request[1], query_dt, body_yaw=body_yaw), request[1]

            self.braking += 1
            _, _, gap, _ = self.ground.foot_state(state)
            elapsed = self.braking * CONTROL_DT
            if elapsed < .18 or (elapsed < .65 and not bool((gap < .012).all())):
                axis, sign = self.direction
                velocity = [0., 0., 0.]
                velocity[axis] = sign * self.stick.ranges[self.direction][0]
                return self.planner.step(observation, velocity, query_dt, body_yaw=body_yaw), velocity

            self._capture_stance(state)
            self.direction = None
            self.settling = round(.30 / CONTROL_DT)

        local, heading = self._standing_targets(state)
        if request is not None and not self.settling:
            self.direction, velocity = request
            self.planner.reset(local, heading, body_yaw=body_yaw)
            return self.planner.step(observation, velocity, query_dt, body_yaw=body_yaw), velocity

        self.settling = max(0, self.settling-1)
        plan = torch.cat((observation.new_tensor([[query_dt, 1., 1.]]),
                          local.flatten(1), heading.flatten(1)), dim=1)
        return plan, (0., 0., 0.)
