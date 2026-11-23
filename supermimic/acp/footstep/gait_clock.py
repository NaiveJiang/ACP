"""Event-calibrated clock for constant forward commands in the phase-x experiment."""
import json
from pathlib import Path

import numpy as np


def calibrate(parts, *, individual_cycles=False, with_events=False, omnidirectional=False):
    rows = []
    for part in parts:
        if part["case"] == "continuous" or (not omnidirectional and part["category"] != "forward"):
            continue
        source = Path(part["source"])
        threshold = json.loads((source / "manifest.json").read_text())["force_threshold_N"]
        with np.load(source / (part["case"] + "_labels.npz")) as data:
            window = slice(part["start"], part["end"])
            velocity = data["velocity_command"][window]
            feet = data["foot_state"][window]
        if omnidirectional:
            magnitude = np.linalg.norm(velocity, axis=1)
            command = velocity[magnitude.argmax()]
            moving = np.flatnonzero(magnitude > 1e-6)
            plateau = np.flatnonzero(np.isclose(velocity,command,atol=1e-6).all(axis=1))
            if not len(moving) or magnitude[0] > 1e-6:
                continue
            speed = tuple(float(x) for x in command)
        else:
            vx = velocity[:,0]
            speed = float(vx.max())
            if speed <= 0 or vx[0] != 0:
                continue
            moving = np.flatnonzero(vx > 1e-6)
            plateau = np.flatnonzero(np.isclose(vx,speed))
        contact = (feet[:, :, 13] > 0) & (feet[:, :, 9] >= threshold)
        state = contact[:, 0].astype(int) + 2 * contact[:, 1].astype(int)
        # Require each contact transition to persist for two samples.
        for i in range(1, len(state)-1):
            if state[i-1] == state[i+1]:
                state[i] = state[i-1]
        singles = np.flatnonzero((np.arange(len(state)) >= moving[0]) & np.isin(state, [1, 2]))
        if not len(singles) or state[moving[0]-1] != 3:
            continue
        left = (state & 1) != 0
        strikes = np.flatnonzero(left[1:] & ~left[:-1]) + 1
        cycles = []
        starts_of_cycles = []
        for a, b in zip(strikes[:-1], strikes[1:]):
            if a < plateau[0] or b > plateau[-1]:
                continue
            segment = state[a:b]
            starts = np.r_[0, np.flatnonzero(np.diff(segment)) + 1]
            if list(segment[starts]) == [3, 1, 3, 2]:
                starts_of_cycles.append(a)
                cycles.append(np.diff(np.r_[starts, len(segment)]) * .02)
        if cycles:
            durations = cycles if individual_cycles else [np.median(cycles, axis=0)]
            for i, cycle in enumerate(durations):
                row = (speed, (singles[0]-moving[0])*.02, int(state[singles[0]]), cycle)
                if with_events:
                    if not individual_cycles:
                        raise ValueError("Event indices require individual cycles")
                    row += (part, int(moving[0]), int(starts_of_cycles[i]))
                rows.append(row)
    if not rows:
        raise ValueError("No complete walking cycles with a standing start")
    return sorted(rows, key=lambda r:r[0])


class ForwardClock:
    def __init__(self, rows, speed):
        speeds = [r[0] for r in rows]
        if speed < speeds[0]-1e-6 or speed > speeds[-1]+1e-6:
            raise ValueError(f"Clock speed outside calibrated range {min(speeds)}..{max(speeds)}")
        self.delay = np.interp(speed, speeds, [r[1] for r in rows])
        durations = [np.interp(speed, speeds, [r[3][j] for r in rows]) for j in range(4)]
        self.ends = np.cumsum(durations)
        self.period = self.ends[-1]
        first = min(rows, key=lambda r:abs(speed-r[0]))[2]
        self.offset = self.ends[0 if first == 1 else 2]

    def query(self, elapsed, horizon):
        time = elapsed + horizon - self.delay
        phase_time = (self.offset + time) % self.period
        stage = np.searchsorted(self.ends, phase_time, side="right") % 4
        support = (3, 1, 3, 2)[stage] if time >= 0 else 3
        return phase_time / self.period, support, time < 0
