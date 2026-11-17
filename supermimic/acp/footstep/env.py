"""Independent two-foot environment using ACP's tested mjlab physics backend."""
from ..env import TrackingSim


class FootstepTrackingSim(TrackingSim):
    """FootstepMotion reserves future target tails; FootstepMDP supplies new obs/reward.

    Does not change the legacy environment or its nine-body task.
    """
    pass
