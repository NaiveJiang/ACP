"""Per-frame contact labels of the 9 contact bodies (feet, knees, hands, elbows, pelvis) for every clip and terrain
variant of the PHP parkour motion set.

The clips store the root pose and joint angles, so the bodies come from forward kinematics. Each frame is replayed
kinematically in MuJoCo on the G1 MJCF inside the clip's terrain: the ground plane z = 0 and the boxes of one variant
(:func:`supermimic.acp.terrain.load_clip_boxes`). A body's contact distance is the smallest signed distance from its
collision geoms (sole spheres, convex limb hulls) to the terrain, from ``mj_geomDistance`` (checked,
:func:`geom_distance`). Contact = distance <= ``--threshold``. Collision geoms of no contact body (hips, thighs,
torso, head, shoulders, wrist roll / pitch links) are reported as ``other_contact``.

Output, ``<out>/<folder>/<clip>.npz`` (outside the motion folders, whose glob is the motion set); V = the clip's
terrain variants (1 for clips without boxes):
    body_names (9,), link_names (9,), fps
    contact_info      (V, T, 9, 1) float32  1 = in contact
    body_pos_w        (T, 9, 3)             target link position
    body_quat_w       (T, 9, 4)             target link orientation, wxyz
    contact_dist      (V, T, 9)             signed distance to the terrain (m, < 0 penetrating), capped at --max_dist
    contact_pos_w     (V, T, 9, 3)          nearest terrain point (NaN beyond --max_dist)
    contact_normal_w  (V, T, 9, 3)          unit normal from that point toward the body (NaN beyond --max_dist)
    contact_surface   (V, T, 9) int8        -1 nothing within --max_dist, 0 ground, k >= 1 box ``boxes[v, k - 1]``
    other_dist        (V, T, L)             as contact_dist for the links of no body, ``other_link_names`` (L,)
    other_contact     (V, T) bool           one of them is in contact
    boxes             (V, K, 9)             the clip's boxes, ``terrain.BOX_FIELDS`` columns
plus ``summary.json`` (per folder and body, on each clip's first variant: contact rate, ground / box top / side / edge
split, segments).

    python -m supermimic.acp.make_labels            # data/php_parkour -> data/php_parkour_contacts
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from multiprocessing import Pool

import mujoco
import numpy as np

from supermimic.acp.robot import CONTACT_BODIES, MOTION_DIR, MOTION_GLOB, get_spec
from supermimic.acp.robot import contact_geom_ids, ROBOT_ID
from supermimic.acp.geometry import primitive_surface_points
from supermimic.acp.motion import load_php_npz
from supermimic.acp.terrain import load_terrain
from supermimic.acp import heightfield


def build_model(boxes: np.ndarray, *, collisions=False, margin=0.) -> mujoco.MjModel:
    """The robot with the ground plane and ``boxes`` (rows with zero size are padding)."""
    spec = get_spec()
    wb = spec.worldbody
    wb.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 1], contype=int(collisions), conaffinity=int(collisions),margin=margin)
    for k, row in enumerate(boxes):
        cx, cy, top, c, s, hx, hy, hz = row[:8]
        if len(row)>8 and row[8]>0:
            heightfield.add_geom(spec,int(row[8])-1,f"box{k}",collision=collisions,margin=margin)
            continue
        if hx > 0:
            yaw = np.arctan2(s, c)
            wb.add_geom(name=f"box{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, hz], pos=[cx, cy, top - hz],
                        quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)], contype=int(collisions), conaffinity=int(collisions),margin=margin)  # fmt: skip
    return spec.compile()


def box_normal(p: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Outward normal of the face of ``box`` (a BOX_FIELDS row) that robot point ``p`` is most outside of
    (inside: nearest to); the fallback when the witness points coincide."""
    cx, cy, top, c, s, hx, hy, hz = box[:8]
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    q = R.T @ (p - np.array([cx, cy, top - hz]))
    i = int(np.argmax(np.abs(q) - np.array([hx, hy, hz])))
    return np.sign(q[i]) * R[:, i]


def surface_points(m: mujoco.MjModel, g: int) -> np.ndarray:
    """Points on the surface of geom g, in its frame: a mesh's vertices, or a sampled box surface."""
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        i = m.geom_dataid[g]
        return m.mesh_vert[m.mesh_vertadr[i] : m.mesh_vertadr[i] + m.mesh_vertnum[i]].astype(np.float64)
    return primitive_surface_points(m.geom_type[g], m.geom_size[g])


def geom_distance(m: mujoco.MjModel, d: mujoco.MjData, g: int, tg: int, max_dist: float, fromto: np.ndarray) -> float:
    """``mj_geomDistance`` from robot geom g to terrain geom tg, checked: near a box edge its GJK can return exactly 0
    with a witness point far outside geom g; then the distance comes from points on g's surface."""
    dist = mujoco.mj_geomDistance(m, d, g, tg, max_dist, fromto)
    if np.linalg.norm(fromto[:3] - d.geom_xpos[g]) <= m.geom_rbound[g] + 1e-4 or dist >= max_dist:
        return dist
    if m.geom_type[tg] != mujoco.mjtGeom.mjGEOM_BOX:
        raise RuntimeError(f"mj_geomDistance degenerate for {m.geom(g).name} / {m.geom(tg).name}")
    x = d.geom_xpos[g] + surface_points(m, g) @ d.geom_xmat[g].reshape(3, 3).T
    q = (x - d.geom_xpos[tg]) @ d.geom_xmat[tg].reshape(3, 3)
    e = np.abs(q) - m.geom_size[tg]
    sdf = np.linalg.norm(np.maximum(e, 0), axis=-1) + np.minimum(e.max(-1), 0)
    i = int(sdf.argmin())
    if sdf[i] >= max_dist:
        return max_dist
    on_box = np.sign(q[i]) * np.minimum(np.abs(q[i]), m.geom_size[tg])  # nearest box point (outside the box)
    fromto[:3], fromto[3:] = x[i], d.geom_xpos[tg] + d.geom_xmat[tg].reshape(3, 3) @ on_box
    return float(sdf[i])


def variant_labels(args) -> dict:
    """Labels of one clip in one terrain variant."""
    path, boxes, threshold, max_dist, *profiles = args
    profile = profiles[0] if profiles else "legacy"
    if boxes.shape[-1]>8 and np.any(boxes[:,8]>0):
        if profile != "legacy":
            raise ValueError("Extended contact profile currently supports box terrain only")
        return heightfield_labels(path,boxes,threshold,max_dist)
    boxes = boxes[boxes[:, 5] > 0]
    m = build_model(boxes)
    d = mujoco.MjData(m)
    pos, quat, jp, fps = load_php_npz(path)
    T, B = len(pos), len(CONTACT_BODIES)

    terrain = {m.geom("ground").id} | {m.geom(f"box{k}").id for k in range(len(boxes))}
    col = np.array([g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and g not in terrain])
    col_body = m.geom_bodyid[col]
    owner = np.full(len(col), -1)  # index into CONTACT_BODIES, -1: other
    for b, ids in enumerate(contact_geom_ids(m, profile)):
        owner[np.isin(col, ids)] = b
    other_bodies = sorted(set(col_body[owner < 0].tolist()))
    other_of = np.array([other_bodies.index(bid) if o < 0 else -1 for bid, o in zip(col_body, owner)])
    frame_ids = [m.body(link).id for link, _ in CONTACT_BODIES.values()]
    ground = m.geom("ground").id
    box_ids = np.array([m.geom(f"box{k}").id for k in range(len(boxes))], dtype=int)
    rb = m.geom_rbound[col]
    box_c = np.c_[boxes[:, :2], boxes[:, 2] - boxes[:, 7]] if len(boxes) else np.zeros((0, 3))
    box_rb = m.geom_rbound[box_ids] if len(boxes) else np.zeros(0)

    out = {
        "contact_dist": np.full((T, B), max_dist, np.float32),
        "contact_pos_w": np.full((T, B, 3), np.nan, np.float32),
        "contact_normal_w": np.full((T, B, 3), np.nan, np.float32),
        "contact_surface": np.full((T, B), -1, np.int8),
        "body_pos_w": np.zeros((T, B, 3), np.float32),
        "body_quat_w": np.zeros((T, B, 4), np.float32),
    }
    other_dist = np.full((T, len(other_bodies)), max_dist, np.float32)
    fromto = np.zeros(6)
    for t in range(T):
        d.qpos[:3], d.qpos[3:7], d.qpos[7:] = pos[t], quat[t], jp[t]
        mujoco.mj_kinematics(m, d)
        out["body_pos_w"][t] = d.xpos[frame_ids]
        out["body_quat_w"][t] = d.xquat[frame_ids]
        gp = d.geom_xpos[col]
        near_ground = gp[:, 2] - rb < max_dist
        near_box = np.linalg.norm(gp[:, None] - box_c[None], axis=-1) - rb[:, None] - box_rb[None] < max_dist  # (G, K)
        for i in range(len(col)):
            cands = ([(ground, 0)] if near_ground[i] else []) + [(box_ids[k], k + 1) for k in np.flatnonzero(near_box[i])]
            b = owner[i]
            for tg, surf in cands:
                dist = geom_distance(m, d, int(col[i]), int(tg), max_dist, fromto)
                if b < 0:
                    other_dist[t, other_of[i]] = min(other_dist[t, other_of[i]], dist)
                elif dist < out["contact_dist"][t, b]:
                    out["contact_dist"][t, b] = dist
                    out["contact_surface"][t, b] = surf
                    out["contact_pos_w"][t, b] = fromto[3:]
                    if surf == 0:
                        out["contact_normal_w"][t, b] = (0.0, 0.0, 1.0)
                    elif abs(dist) > 1e-6:  # witness points: the face normal on a face, in between on an edge
                        out["contact_normal_w"][t, b] = (fromto[:3] - fromto[3:]) / dist
                    else:
                        out["contact_normal_w"][t, b] = box_normal(fromto[:3], boxes[surf - 1])
    out["other_dist"] = other_dist
    out["other_contact"] = (other_dist <= threshold).any(1)
    out["contact_info"] = (out["contact_dist"] <= threshold).astype(np.float32)[..., None]
    out["other_link_names"] = np.array([m.body(b).name for b in other_bodies])
    out["fps"] = np.float32(fps)
    return out


def heightfield_labels(path, boxes, threshold, max_dist):
    """Surface-sampled robot geometry against exact local heightfield triangles.

    MuJoCo heightfield distance queries and nonzero collision margins do not
    provide reliable separation witnesses. Sample primitive surfaces at 4 mm,
    then resolve the eight nearest candidates against the field triangles.
    Plane fixtures and source/label penetration gates bound this approximation.
    """
    import torch
    torch.set_num_threads(1)
    m=build_model(boxes);d=mujoco.MjData(m)
    pos,quat,jp,fps=load_php_npz(path);T,B=len(pos),len(CONTACT_BODIES)
    terrain={m.geom("ground").id}|{m.geom(f"box{k}").id for k in range(len(boxes)) if boxes[k,5]>0}
    groups=contact_geom_ids(m);owner={g:b for b,gs in enumerate(groups) for g in gs}
    col=[g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and g not in terrain]
    other_bodies=sorted({int(m.geom_bodyid[g]) for g in col if g not in owner})
    other_of={g:other_bodies.index(int(m.geom_bodyid[g])) for g in col if g not in owner}
    frames=[m.body(link).id for link,_ in CONTACT_BODIES.values()]
    out=dict(contact_dist=np.full((T,B),max_dist,np.float32),contact_pos_w=np.full((T,B,3),np.nan,np.float32),
             contact_normal_w=np.full((T,B,3),np.nan,np.float32),contact_surface=np.full((T,B),-1,np.int16),
             body_pos_w=np.zeros((T,B,3),np.float32),body_quat_w=np.zeros((T,B,4),np.float32),
             other_dist=np.full((T,len(other_bodies)),max_dist,np.float32),
             other_link_names=np.array([m.body(b).name for b in other_bodies]),fps=np.float32(fps))
    centers=np.empty((T,len(col),3));rotations=np.empty((T,len(col),3,3))
    for t in range(T):
        d.qpos[:3],d.qpos[3:7],d.qpos[7:]=pos[t],quat[t],jp[t]
        mujoco.mj_kinematics(m,d)
        out["body_pos_w"][t]=d.xpos[frames];out["body_quat_w"][t]=d.xquat[frames]
        centers[t]=d.geom_xpos[col];rotations[t]=d.geom_xmat[col].reshape(-1,3,3)
    index=int(boxes[0,8])-1;a=heightfield.ASSETS[index];h=a["heights"];ny,nx=h.shape
    def surface(xy):
        uv=(xy-a["origin"])/a["spacing"];u,v=uv[...,0],uv[...,1]
        ix=np.clip(np.floor(u).astype(int),0,nx-2);iy=np.clip(np.floor(v).astype(int),0,ny-2)
        tx,ty=u-ix,v-iy;h00,h10,h01,h11=h[iy,ix],h[iy,ix+1],h[iy+1,ix],h[iy+1,ix+1]
        gx=np.where(tx>=ty,h10-h00,h11-h01)/a["spacing"][0]
        gy=np.where(tx>=ty,h11-h10,h01-h00)/a["spacing"][1]
        z=np.where(tx>=ty,h00+(h10-h00)*tx+(h11-h10)*ty,h00+(h11-h01)*tx+(h01-h00)*ty)
        return z,gx,gy
    for ci,g in enumerate(col):
        local=surface_points(m,g) if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH else primitive_surface_points(m.geom_type[g],m.geom_size[g],spacing=.004)
        for begin in range(0,T,64):
            end=min(T,begin+64);world=centers[begin:end,ci,None]+np.einsum('tij,pj->tpi',rotations[begin:end,ci],local)
            z,gx,gy=surface(world[...,:2]);approx=(world[...,2]-z)/np.sqrt(1+gx*gx+gy*gy)
            count=min(8,len(local));chosen=np.argpartition(approx,count-1,axis=1)[:,:count]
            points=np.take_along_axis(world,chosen[...,None],axis=1).astype(np.float32)
            with torch.no_grad():
                pts=torch.from_numpy(points);idx=torch.full((len(pts),),index,dtype=torch.long)
                dist,witness=heightfield.distance_point(pts,idx)
                ground=pts[...,2];on_ground=ground<dist
                witness=torch.where(on_ground[...,None],torch.cat((pts[...,:2],torch.zeros_like(pts[...,2:])),dim=-1),witness)
                dist=torch.minimum(dist,ground)
                which=dist.argmin(-1);batch=torch.arange(len(pts));minimum=dist[batch,which].numpy()
                cp=witness[batch,which].numpy();robot_pt=points[np.arange(len(points)),which.numpy()]
            times=np.arange(begin,end)
            if g in other_of:
                j=other_of[g];out["other_dist"][times,j]=np.minimum(out["other_dist"][times,j],minimum)
            else:
                b=owner[g];better=minimum<out["contact_dist"][times,b];times=times[better]
                if not len(times):continue
                dd=minimum[better];normal=(robot_pt[better]-cp[better])/np.where(np.abs(dd)>1e-6,dd,1.)[:,None]
                tiny=np.abs(dd)<=1e-6
                if tiny.any():
                    _,gx,gy=surface(cp[better][tiny,:2]);nn=np.stack((-gx,-gy,np.ones_like(gx)),-1);normal[tiny]=nn/np.linalg.norm(nn,axis=-1,keepdims=True)
                out["contact_dist"][times,b]=dd;out["contact_pos_w"][times,b]=cp[better]
                out["contact_normal_w"][times,b]=normal;out["contact_surface"][times,b]=1
    out["other_contact"]=(out["other_dist"]<=threshold).any(1)
    out["contact_info"]=(out["contact_dist"]<=threshold).astype(np.float32)[...,None]
    return out


PER_VARIANT = ("contact_info", "contact_dist", "contact_pos_w", "contact_normal_w", "contact_surface", "other_dist", "other_contact")


def segments(c: np.ndarray) -> np.ndarray:
    """Lengths (frames) of the runs of True in the 1-D bool array ``c``."""
    e = np.diff(np.r_[0, c.astype(np.int8), 0])
    return np.flatnonzero(e == -1) - np.flatnonzero(e == 1)


def summarize(results: list[tuple[str, dict]], threshold: float) -> dict:
    """Per folder (and "all") and body, on each clip's first variant: % of frames in contact, its ground / box top /
    side / edge split, contact segments per clip, and the share of segments of at most 2 frames (flicker)."""
    by_folder: dict[str, list[dict]] = {}
    for rel, r in results:
        by_folder.setdefault(rel.split("/")[0], []).append(r)
    by_folder["all"] = [r for _, r in results]
    summary = {}
    for folder, rs in by_folder.items():
        c = np.concatenate([r["contact_info"][0, ..., 0] > 0.5 for r in rs])
        surf = np.concatenate([r["contact_surface"][0] for r in rs])
        nz = np.concatenate([r["contact_normal_w"][0, ..., 2] for r in rs])
        rows = {"clips": len(rs), "frames": len(c), "other_contact_pct": round(100 * float(np.concatenate([r["other_contact"][0] for r in rs]).mean()), 2)}
        names = sorted({str(n) for r in rs for n in r["other_link_names"]})
        rows["other_links_pct"] = {}
        for n in names:
            hit = [(r["other_dist"][0][:, list(r["other_link_names"]).index(n)] <= threshold) if n in r["other_link_names"] else np.zeros(r["other_dist"].shape[1], bool) for r in rs]
            rows["other_links_pct"][n] = round(100 * float(np.concatenate(hit).mean()), 2)
        for b, name in enumerate(CONTACT_BODIES):
            cb = c[:, b]
            segs = [segments(r["contact_info"][0, :, b, 0] > 0.5) for r in rs]
            segs_all = np.concatenate(segs) if segs else np.zeros(0)
            n = max(int(cb.sum()), 1)
            rows[name] = {
                "contact_pct": round(100 * float(cb.mean()), 2),
                "ground_pct": round(100 * float((cb & (surf[:, b] == 0)).sum()) / n, 1),
                "box_top_pct": round(100 * float((cb & (surf[:, b] > 0) & (nz[:, b] > 0.95)).sum()) / n, 1),
                "box_side_pct": round(100 * float((cb & (surf[:, b] > 0) & (np.abs(nz[:, b]) < 0.05)).sum()) / n, 1),
                "box_edge_pct": round(100 * float((cb & (surf[:, b] > 0) & (nz[:, b] <= 0.95) & (np.abs(nz[:, b]) >= 0.05)).sum()) / n, 1),
                "segments_per_clip": round(len(segs_all) / len(rs), 1),
                "short_segments_pct": round(100 * float((segs_all <= 2).mean()), 1) if len(segs_all) else 0.0,
            }
        summary[folder] = rows
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motion_dir", default=MOTION_DIR)
    ap.add_argument("--out", default=None, help="default: <motion_dir>_contacts")
    ap.add_argument("--threshold", type=float, default=0.005, help="contact if the signed distance is at most this (m)")
    ap.add_argument("--contact-profile", choices=("legacy", "roban_knee_caps_v1"), default="legacy")
    ap.add_argument("--max_dist", type=float, default=0.1, help="distances, terrain points and normals beyond this are not resolved (m)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--clips", type=str, default=None, help="only clips whose name contains this (testing)")
    args = ap.parse_args()
    out_dir = args.out or args.motion_dir.rstrip("/") + "_contacts"

    files = sorted(glob.glob(os.path.join(args.motion_dir, MOTION_GLOB)))
    if args.clips:
        files = [f for f in files if args.clips in f]
    terrains = [load_terrain(f) for f in files]
    tasks = [(f, b, args.threshold, args.max_dist, args.contact_profile) for f, tv in zip(files, terrains) for b in tv]
    t0 = time.time()
    with Pool(args.workers,initializer=heightfield.initialize,initargs=([a["path"] for a in heightfield.ASSETS],)) as pool:
        res = pool.map(variant_labels, tasks, chunksize=1)
    results, i = [], 0
    for f, tv in zip(files, terrains):
        rs, i = res[i : i + len(tv)], i + len(tv)
        r = {k: np.stack([x[k] for x in rs]) for k in PER_VARIANT}
        r.update(robot_id=np.array(ROBOT_ID), body_pos_w=rs[0]["body_pos_w"], body_quat_w=rs[0]["body_quat_w"], other_link_names=rs[0]["other_link_names"],
                 fps=rs[0]["fps"], boxes=tv.astype(np.float32), threshold=np.float32(args.threshold),
                 body_names=np.array(list(CONTACT_BODIES)), link_names=np.array([v[0] for v in CONTACT_BODIES.values()]))  # fmt: skip
        r['terrain_label_method']=np.array('surface_samples_4mm_local_heightfield_triangles' if np.any(tv[...,8]>0) else 'mujoco_geom_distance')
        r['contact_profile'] = np.array(args.contact_profile)
        rel = os.path.relpath(f, args.motion_dir)[: -len("_motion.npz")] + ".npz"
        os.makedirs(os.path.join(out_dir, os.path.dirname(rel)), exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, rel), source_motion=rel, **r)
        results.append((rel, r))
    summary = summarize(results, args.threshold)
    summary["config"] = {"threshold": args.threshold, "max_dist": args.max_dist, "contact_profile": args.contact_profile,
                         "bodies": {k: {"frame": v[0], "geoms_of": list(v[1])} for k, v in CONTACT_BODIES.items()}}  # fmt: skip
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"{len(files)} clips, {len(tasks)} clip variants, {summary['all']['frames']} frames in {time.time() - t0:.0f} s -> {out_dir}")
    print(f"\n{'contact % of frames':24s}" + "".join(f"{k[:12]:>13s}" for k in CONTACT_BODIES) + f"{'other':>8s}")
    for folder, rows in summary.items():
        if folder != "config":
            print(f"{folder:24s}" + "".join(f"{rows[k]['contact_pct']:13.1f}" for k in CONTACT_BODIES) + f"{rows['other_contact_pct']:8.1f}")
    a = summary["all"]
    print(f"\n{'all clips':24s}{'ground':>8s}{'box top':>9s}{'box side':>10s}{'box edge':>10s}{'seg/clip':>10s}{'<=2 fr':>8s}  (% of contact frames / segments)")
    for k in CONTACT_BODIES:
        r = a[k]
        print(f"{k:24s}{r['ground_pct']:8.1f}{r['box_top_pct']:9.1f}{r['box_side_pct']:10.1f}{r['box_edge_pct']:10.1f}{r['segments_per_clip']:10.1f}{r['short_segments_pct']:8.1f}")
    print("\nother links in contact (% of frames): " + ", ".join(f"{n} {p}" for n, p in a["other_links_pct"].items() if p > 0))


if __name__ == "__main__":
    main()
