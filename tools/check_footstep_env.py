"""Validate current contact training dependencies without starting training."""
import argparse
import ctypes
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError('Current contact environment requires Python 3.11')
    os.environ['ACP_ROBOT'] = 'roban'
    os.environ.setdefault('MUJOCO_GL', 'egl')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    expected = {'torch': '2.9.0', 'torchvision': '0.24.0', 'numpy': '2.3.4',
                'mujoco': '3.11.0', 'mujoco-warp': '3.11.0',
                'warp-lang': '1.14.0', 'mjlab': '1.6.0'}
    versions = {name: version(name) for name in expected}
    for name, want in expected.items():
        if versions[name].split('+')[0] != want:
            raise RuntimeError(f'{name}: found {versions[name]}, expected {want}')
    for name in ('torchvision', 'numpy', 'scipy', 'mujoco', 'mjlab', 'warp',
                 'matplotlib', 'PIL', 'imageio', 'imageio_ffmpeg', 'yaml',
                 'supermimic.acp.env', 'supermimic.acp.footstep.fast.train',
                 'supermimic.acp.footstep.stable_contact'):
        importlib.import_module(name)
    import torch
    import mujoco
    import warp as wp
    for lib in ('libEGL.so.1', 'libGL.so.1'):
        try:
            ctypes.CDLL(lib)
        except OSError as exc:
            raise RuntimeError(f'Missing {lib}; run bash install_env.sh --system-deps') from exc
    gpu = None
    if args.device == 'cuda':
        if torch.version.cuda != '12.8' or not torch.cuda.is_available():
            raise RuntimeError(f'CUDA 12.8 build/GPU unavailable: torch CUDA={torch.version.cuda}; check NVIDIA driver')
        gpu = torch.cuda.get_device_name(0)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('Default fast training requires BF16 support; this GPU is unsupported by the default profile')
        wp.init()
        if not wp.is_cuda_available():
            raise RuntimeError('Warp cannot access CUDA')
        # Check actual GPU operations and graph replay, not just package metadata.
        x = torch.ones(16, 16, device='cuda', requires_grad=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = (x@x).square().mean()
        loss.backward()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = x.detach()*2
        graph.replay()
        torch.cuda.synchronize()
        if not torch.isfinite(x.grad).all() or not (output == 2).all():
            raise RuntimeError('CUDA backward/graph validation failed')
    asset = root/'data/roban_description/xml/biped_s17_verified_handball_fixedhead.xml'
    assets_ok = asset.is_file()
    if assets_ok:
        from supermimic.acp.robot import get_spec
        get_spec().compile()
    else:
        print('机器人资产尚未复制：请另外恢复 data/roban_description/；依赖安装不包含训练数据。')
    # A minimal EGL render catches headless graphics failures even without assets.
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=32, width=32) as renderer:
        renderer.update_scene(data)
        if renderer.render().shape != (32, 32, 3):
            raise RuntimeError('Headless render failed')
    report = dict(python=sys.version.split()[0], executable=sys.executable, versions=versions,
                  device=args.device, gpu=gpu, robot_assets=assets_ok, headless_render=True)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    print('依赖与运行检查通过。')


if __name__ == '__main__':
    main()
