"""Select an ACP robot before loading any dimension-dependent module."""
import argparse
import os
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("g1", "roban"), default=os.environ.get("ACP_ROBOT", "g1"))
    parser.add_argument("command", choices=("train_stage1", "train_stage2", "train_stage3", "train_planner_offline",
        "evaluate", "evaluate_goal", "make_labels", "refine_motion", "render", "render_labels",
        "demo_reference", "prepare_motion", "smoke_robot", "interactive"))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    os.environ["ACP_ROBOT"] = args.robot
    module = f"supermimic.acp.{args.command}"
    sys.argv = [module, *args.args]
    # Multiprocessing workers must be able to resolve functions defined by the
    # selected module (e.g. make_labels.variant_labels) through __main__.
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
