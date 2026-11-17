"""Leakage-safe v2 contact data, compact mmap cache and five-frame observations."""
from __future__ import annotations

import bisect
from collections import OrderedDict, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from torch.utils.data import Dataset


JOINT_NAMES = (["waist_yaw_joint"]
               + [f"leg_{side}{i}_joint" for side in "lr" for i in range(1, 7)]
               + [f"zarm_{side}{i}_joint" for side in "lr" for i in range(1, 5)])
STATE_FIELDS = ["gravity_body[3]", "angular_velocity_body[3]", "joint_position[21]", "joint_velocity[21]"]
STATE_DIM = 48
CACHE_VERSION = 1
ARRAYS = ("state", "velocity", "dt", "contact", "position", "heading")


def observations(qpos, qvel, metadata):
    """MuJoCo free-joint qvel angular components are already in the root frame."""
    joints = {j["name"]: j for j in metadata["joints"]}
    qp = [joints[n]["qpos_adr"] for n in JOINT_NAMES]
    qv = [joints[n]["dof_adr"] for n in JOINT_NAMES]
    w, x, y, z = qpos[:, 3:7].T
    if not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=-1), 1., atol=1e-3):
        raise ValueError("Expected unit wxyz root quaternions")
    gravity = -np.stack((2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), axis=-1)
    result = np.concatenate((gravity, qvel[:, 3:6], qpos[:, qp], qvel[:, qv]), axis=-1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite proprioceptive observation")
    return result


def prepare(sources, cache_dir):
    """Unpack only required arrays once, keyed by file/metadata fingerprint."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    common_horizons = None
    seen_sources = set()
    for source in sources:
        source = Path(source).resolve()
        if not (source / "manifest.json").exists():
            source = source / "future_contact_v2"
        if str(source) in seen_sources:
            raise ValueError("Duplicate dataset: " + str(source))
        seen_sources.add(str(source))
        manifest = json.loads((source / "manifest.json").read_text())
        if manifest.get("schema_version") != 2 or manifest.get("status") != "complete":
            raise ValueError("Need complete future_contact_v2 dataset: " + str(source))
        horizons = manifest["horizons_s"]
        if common_horizons is None:
            common_horizons = horizons
        elif common_horizons != horizons:
            raise ValueError("All datasets must have the same horizons")
        metadata_text = (source.parent / "metadata.json").read_text()
        metadata = json.loads(metadata_text)
        for info in manifest["clips"]:
            name = info["case"]
            if Path(name).name != name or name in (".", ".."):
                raise ValueError("Unsafe case name")
            path = source / (name + "_labels.npz")
            stat = path.stat()
            fingerprint = json.dumps([str(path), stat.st_size, stat.st_mtime_ns, metadata_text,
                                      CACHE_VERSION, JOINT_NAMES])
            key = hashlib.sha256(fingerprint.encode()).hexdigest()[:24]
            target = cache_dir / key
            if not (target / "info.json").exists():
                temp = Path(tempfile.mkdtemp(prefix=key + ".", dir=cache_dir))
                try:
                    with np.load(path, allow_pickle=False) as d:
                        state = observations(d["qpos"], d["qvel"], metadata)
                        arrays = dict(state=state, velocity=d["velocity_command"].astype(np.float32),
                            dt=d["time_to_target"].astype(np.float32), contact=d["future_contact"].astype(np.float32),
                            position=d["target_pos_base"].astype(np.float32),
                            heading=d["target_yaw_base_sincos"].astype(np.float32),
                            valid=d["planner_training_valid"], future_index=d["future_index"])
                        categories = sorted(set(d["category"].tolist()))
                        if arrays["valid"].shape != (len(state), len(horizons)):
                            raise ValueError("Unexpected target shape: " + str(path))
                        if not np.array_equal(arrays["valid"], d["training_valid"] & d["command_window_constant"]):
                            raise ValueError("Inconsistent planner validity mask: " + str(path))
                    for field, arr in arrays.items():
                        np.save(temp / (field + ".npy"), arr, allow_pickle=False)
                    descriptor = dict(source=str(source), case=name, category="+".join(categories),
                                      frames=len(state), horizons=horizons, cache=str(target.resolve()),
                                      fingerprint=key, fps=float(metadata.get("sample_hz", 50.)))
                    (temp / "info.json").write_text(json.dumps(descriptor, indent=2))
                    os.rename(temp, target)
                finally:
                    if temp.exists():
                        shutil.rmtree(temp)
            clips.append(json.loads((target / "info.json").read_text()))
    if not clips:
        raise ValueError("No clips")
    return clips


def split_clips(clips, val_fraction=.2, seed=42):
    """Hold out complete clips per source/category; singleton groups use time blocks.

    Every sample's entire history and future target must lie in its assigned block.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0,1)")
    groups = defaultdict(list)
    for clip in clips:
        groups[(clip["source"], clip["category"])].append(clip)
    rng = np.random.default_rng(seed)
    result = {"train": [], "val": []}
    for group in groups.values():
        if len(group) > 1:
            nval = min(len(group)-1, max(1, math.ceil(len(group)*val_fraction)))
            held = set(rng.permutation(len(group))[:nval].tolist())
            for i, clip in enumerate(group):
                result["val" if i in held else "train"].append(dict(clip, start=0, end=clip["frames"]))
        else:
            clip = group[0]
            cut = int(clip["frames"]*(1-val_fraction))
            result["train"].append(dict(clip, start=0, end=cut))
            result["val"].append(dict(clip, start=cut, end=clip["frames"]))
    return result


class ContactDataset(Dataset):
    def __init__(self, intervals, hist=5, cache_clips=16):
        if hist < 1 or cache_clips < 1:
            raise ValueError("hist and cache_clips must be positive")
        self.hist, self.cache_clips = hist, cache_clips
        self.entries, self.ends, self.cache = [], [], OrderedDict()
        count = 0
        for interval in intervals:
            folder = Path(interval["cache"])
            valid = np.load(folder / "valid.npy", mmap_mode="r").copy()
            future = np.load(folder / "future_index.npy", mmap_mode="r")
            frame = np.arange(len(valid))[:, None]
            valid &= (frame >= interval["start"]+hist-1) & (frame < interval["end"])
            valid &= (future >= frame) & (future < interval["end"])
            pairs = np.column_stack(np.nonzero(valid)).astype(np.int32)
            if len(pairs):
                self.entries.append((interval, pairs))
                count += len(pairs)
                self.ends.append(count)
        if not count:
            raise ValueError("No valid samples after history/split filtering")

    def __len__(self):
        return self.ends[-1]

    def locate(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        k = bisect.bisect_right(self.ends, index)
        interval, pairs = self.entries[k]
        i, h = pairs[index - (self.ends[k-1] if k else 0)]
        return interval, int(i), int(h)

    def __getitem__(self, index):
        interval, i, h = self.locate(index)
        key = interval["cache"]
        if key not in self.cache:
            self.cache[key] = {n: np.load(Path(key)/(n+".npy"), mmap_mode="r") for n in ARRAYS}
            if len(self.cache) > self.cache_clips:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        d = self.cache[key]
        return dict(state_history=d["state"][i-self.hist+1:i+1].copy(),
                    velocity_command=d["velocity"][i].copy(), query_dt=d["dt"][i, h:h+1].copy(),
                    target_contact=d["contact"][i, h].copy(),
                    target_position=d["position"][i, h].copy(),
                    target_heading=d["heading"][i, h].copy(),
                    support_mask=d["contact"][i, h].copy(), horizon_index=np.int64(h))

    def balanced_weights(self):
        """Equal aggregate sampling weight for each category/horizon combination."""
        counts = defaultdict(int)
        for interval, pairs in self.entries:
            hs, ns = np.unique(pairs[:, 1], return_counts=True)
            for h, n in zip(hs, ns):
                counts[(interval["category"], int(h))] += int(n)
        weights = np.empty(len(self), dtype=np.float64)
        start = 0
        for interval, pairs in self.entries:
            for h in np.unique(pairs[:, 1]):
                sel = pairs[:, 1] == h
                weights[start:start+len(pairs)][sel] = 1/counts[(interval["category"], int(h))]
            start += len(pairs)
        return weights

    def summary(self):
        by_category = defaultdict(int)
        for interval, pairs in self.entries:
            by_category[interval["category"]] += len(pairs)
        return dict(samples=len(self), intervals=len(self.entries), by_category=dict(by_category))
