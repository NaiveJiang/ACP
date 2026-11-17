"""Finite heightfield assets shared by scans, contact rewards and MuJoCo.

External sidecar: <clip>_heightfield.npz with heights[ny,nx] (world z),
origin_xy[2] (lower left), spacing[2]. Internal terrain column 8 is a 1-based
process-local asset index; zero denotes a legacy box. Indices are never inputs
to a learned model. MuJoCo and torch use the same lower-left/upper-right split.
"""
from pathlib import Path

import mujoco
import numpy as np
import torch

ASSETS = []
_paths = {}
_cache = {}


def register(path):
    key = str(Path(path).resolve())
    if key not in _paths:
        with np.load(key, allow_pickle=False) as z:
            h = z["heights"].astype(np.float32)
            origin, spacing = z["origin_xy"].astype(float), z["spacing"].astype(float)
        if (h.ndim != 2 or min(h.shape) < 2 or not np.isfinite(h).all()
                or origin.shape != (2,) or spacing.shape != (2,)
                or not np.isfinite(origin).all() or not np.isfinite(spacing).all()
                or np.any(spacing <= 0) or h.min() < -1e-5):
            raise ValueError(f"Invalid heightfield: {key}")
        ny, nx = h.shape
        half = spacing*np.array([nx-1, ny-1])/2
        _paths[key] = len(ASSETS)
        ASSETS.append(dict(path=key, heights=h, origin=origin, spacing=spacing,
                           center=origin+half, half=half, low=float(h.min()), high=float(h.max())))
        _cache.clear()
    return _paths[key]


def token(index):
    a = ASSETS[index]
    return np.array([*a["center"], a["high"], 1., 0., *a["half"], max(.001, (a["high"]-a["low"])/2), index+1])


def initialize(paths):
    """Multiprocessing initializer; preserves the parent's asset index order."""
    ASSETS.clear(); _paths.clear(); _cache.clear()
    for path in paths:
        register(path)


def add_asset(spec, index, name):
    a = ASSETS[index]
    h = a["heights"]
    scale = max(1e-5, a["high"]-a["low"])
    spec.add_hfield(name=name, nrow=h.shape[0], ncol=h.shape[1],
                    size=[*a["half"], scale, max(.1, a["low"]+.1)],
                    userdata=((h-a["low"])/scale).ravel().tolist())


def add_geom(spec, index, name, *, collision=True, margin=0.):
    a = ASSETS[index]
    add_asset(spec, index, name)
    return spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=name,
        pos=[*a["center"], a["low"]], contype=int(collision), conaffinity=int(collision),
        margin=margin, rgba=[.45,.55,.42,1.])


def tensors(device):
    key = str(device)
    if key not in _cache:
        metadata, arrays, offset = [], [], 0
        for a in ASSETS:
            ny, nx = a["heights"].shape
            metadata.append([offset,nx,ny,*a["origin"],*a["spacing"]])
            arrays.append(a["heights"].ravel()); offset += nx*ny
        _cache[key] = (torch.tensor(np.array(metadata), dtype=torch.float64, device=device),
                       torch.tensor(np.concatenate(arrays), dtype=torch.float32, device=device))
    return _cache[key]


def _cells(xy, indices):
    meta, values = tensors(xy.device)
    info = meta[indices.clamp(min=0)]
    offset,nx,ny = (info[:,i:i+1] for i in range(3))
    ox,oy,dx,dy = (info[:,i:i+1].to(xy.dtype) for i in range(3,7))
    u, v = (xy[...,0]-ox)/dx, (xy[...,1]-oy)/dy
    # NaN.long() is INT64_MIN: clamping a NaN float does not make it a safe
    # gather index. Keep the original coordinates for validity reporting, but
    # always gather inside this asset (also when the query is outside its grid).
    ix = torch.minimum(torch.nan_to_num(u, nan=0.).floor().clamp(min=0), nx-2).long()
    iy = torch.minimum(torch.nan_to_num(v, nan=0.).floor().clamp(min=0), ny-2).long()
    return info, values, u, v, ix, iy


def heights(xy, indices):
    info,values,u,v,ix,iy = _cells(xy,indices)
    off,nx,ny = (info[:,i:i+1].long() for i in range(3))
    adr=off+iy*nx+ix
    h00,h10,h01,h11 = values[adr],values[adr+1],values[adr+nx],values[adr+nx+1]
    tx,ty=u-ix,v-iy
    h=torch.where(tx>=ty,h00+(h10-h00)*tx+(h11-h10)*ty,
                          h00+(h11-h01)*tx+(h01-h00)*ty)
    h = torch.where((u>=0)&(u<=nx-1)&(v>=0)&(v<=ny-1),h,0.)
    # An invalid state must remain visible to the training numerics guard;
    # interpreting it as flat ground would conceal the upstream failure.
    return torch.where(xy.isfinite().all(-1), h, torch.nan)


def _triangle_point(p,a,b,c):
    ab,ac,pv=b-a,c-a,p-a
    n=torch.linalg.cross(ab,ac)
    proj=p-((pv*n).sum(-1,keepdim=True)/n.square().sum(-1,keepdim=True).clamp_min(1e-20))*n
    q=proj-a
    d00,d01,d11=(ab*ab).sum(-1),(ab*ac).sum(-1),(ac*ac).sum(-1)
    d20,d21=(q*ab).sum(-1),(q*ac).sum(-1)
    den=(d00*d11-d01*d01).clamp_min(1e-20)
    v=(d11*d20-d01*d21)/den;w=(d00*d21-d01*d20)/den
    inside=(v>=0)&(w>=0)&(v+w<=1)
    best=proj;dist=torch.where(inside,(p-proj).square().sum(-1),torch.inf)
    for s,e in ((a,b),(b,c),(c,a)):
        edge=e-s
        t=((p-s)*edge).sum(-1,keepdim=True)/edge.square().sum(-1,keepdim=True).clamp_min(1e-20)
        cand=s+t.clamp(0,1)*edge;dd=(p-cand).square().sum(-1);use=dd<dist
        best=torch.where(use[...,None],cand,best);dist=torch.minimum(dist,dd)
    return best,dist


def distance_point(x,indices):
    """Nearest of the 18 local heightfield triangles, signed below its surface.

    The 3x3 cell stencil resolves local contact distances. Far-field distances
    are conservative upper bounds, not a global mesh SDF; contact gates validate
    the stencil against surface-sampled labels on released references.
    """
    info,values,u,v,ix,iy=_cells(x[...,:2],indices)
    off,nx,ny=(info[:,i:i+1].long() for i in range(3))
    ox,oy,dx,dy=(info[:,i:i+1].to(x.dtype) for i in range(3,7))
    best=x;dd=torch.full_like(x[...,0],torch.inf)
    for sx in (-1,0,1):
        for sy in (-1,0,1):
            j=torch.minimum((ix+sx).clamp(min=0),nx-2);k=torch.minimum((iy+sy).clamp(min=0),ny-2)
            adr=off+k*nx+j
            xx,yy=ox+j*dx,oy+k*dy
            a=torch.stack((xx,yy,values[adr]),-1)
            b=torch.stack((xx+dx,yy,values[adr+1]),-1)
            c=torch.stack((xx+dx,yy+dy,values[adr+nx+1]),-1)
            d=torch.stack((xx,yy+dy,values[adr+nx]),-1)
            for tri in ((a,b,c),(a,c,d)):
                pt,dist=_triangle_point(x,*tri);use=dist<dd
                best=torch.where(use[...,None],pt,best);dd=torch.minimum(dd,dist)
    sign=torch.where(x[...,2]<heights(x[...,:2],indices),-1.,1.)
    return torch.sqrt(dd+1e-12)*sign,best
