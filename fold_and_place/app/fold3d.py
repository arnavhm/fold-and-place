"""3D fold animation: the garment outline becomes a cloth mesh and each FoldStep turns the moving
side over its fold line, like a page of a book.

Geometry is computed here in Python so the browser part stays dumb:

* The mesh is a rectilinear grid clipped to the garment outline. Every fold line is mapped back
  to the flat garment (each point's position is a chain of reflections, so a fold line has one
  pre-image per reflection group) and added as a grid line, so the cloth bends exactly on the
  crease instead of stretching a cell across it.
* Heights: when a stack turns over it lands on whatever lies under its mirror image, one
  fabric gap higher. Turning a stack upside down maps z -> base + top - z, which is a rotation
  by 180 degrees about a hinge at height (base + top) / 2. So each step is fully described by
  (axis, line, hinge height, which vertices move) and the browser just rotates by theta in [0, pi].

The page inlines three.js r147 (MIT, vendored in ``static/``) so it works offline.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
import shapely

from fold_and_place.folding.engine import FoldStep
from fold_and_place.folding.templates import outline
from fold_and_place.garments.record import Garment

STATIC = Path(__file__).parent / "static"
GAP_CM = 0.35  # visual thickness per fabric layer
EPS = 1e-6


def _grid_lines(lo: float, hi: float, n: int, extra: list[float]) -> np.ndarray:
    base = np.linspace(lo, hi, n + 1)
    pts = np.concatenate([base, [e for e in extra if lo + 1e-3 < e < hi - 1e-3]])
    pts = np.unique(np.round(pts, 4))
    # drop slivers (< 0.15 cm) created next to inserted crease lines, keeping crease lines
    keep = [pts[0]]
    extra_set = {round(e, 4) for e in extra}
    for p in pts[1:]:
        if p - keep[-1] < 0.15:
            if round(p, 4) in extra_set and round(keep[-1], 4) not in extra_set and len(keep) > 1:
                keep[-1] = p
            continue
        keep.append(p)
    if keep[-1] < hi - 1e-6:
        keep.append(hi)
    return np.asarray(keep)


def _preimages(xs: np.ndarray, ys: np.ndarray, steps: list[FoldStep]) -> tuple[list[float], list[float]]:
    """Original-coordinate lines that every fold line passes through."""
    X, Y = np.meshgrid(xs, ys)
    px, py = X.ravel().copy(), Y.ravel().copy()
    sx, tx = np.ones_like(px), np.zeros_like(px)  # current x = sx * x0 + tx
    sy, ty = np.ones_like(py), np.zeros_like(py)
    ex, ey = [], []
    for st in steps:
        cur = px if st.axis == "x" else py
        s, t = (sx, tx) if st.axis == "x" else (sy, ty)
        groups = np.unique(np.round(np.stack([s, t], 1), 6), axis=0)
        for gs, gt in groups:
            (ex if st.axis == "x" else ey).append((st.coord - gt) / gs)
        mv = cur < st.coord - EPS if st.move == "lt" else cur > st.coord + EPS
        cur[mv] = 2 * st.coord - cur[mv]
        s[mv] = -s[mv]
        t[mv] = 2 * st.coord - t[mv]
    return ex, ey


def build(g: Garment, steps: list[FoldStep], cells: int = 90) -> dict:
    poly = outline(g)
    x0, y0, x1, y1 = poly.bounds
    size = max(x1 - x0, y1 - y0)
    nx = max(8, int(round(cells * (x1 - x0) / size)))
    ny = max(8, int(round(cells * (y1 - y0) / size)))
    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(y0, y1, ny + 1)
    for _ in range(2):  # second pass catches lines whose groups depended on the first refinement
        ex, ey = _preimages(xs, ys, steps)
        xs = _grid_lines(x0, x1, nx, ex)
        ys = _grid_lines(y0, y1, ny, ey)

    # cells fully inside the outline -> two triangles on shared grid vertices; cells crossing
    # the outline -> clipped to it and triangulated with their own vertices (smooth edges)
    X0, Y0 = np.meshgrid(xs[:-1], ys[:-1])
    X1, Y1 = np.meshgrid(xs[1:], ys[1:])
    boxes = shapely.box(X0, Y0, X1, Y1)
    full = shapely.contains(poly, boxes)
    part = shapely.intersects(poly, boxes) & ~full
    used = np.zeros((len(ys), len(xs)), bool)
    jj, ii = np.nonzero(full)  # row (y), col (x)
    for j, i in zip(jj, ii):
        used[j : j + 2, i : i + 2] = True
    idx = -np.ones(used.shape, int)
    idx[used] = np.arange(used.sum())
    tris: list[int] = []
    for j, i in zip(jj, ii):
        a, b, c, d = idx[j, i], idx[j, i + 1], idx[j + 1, i + 1], idx[j + 1, i]
        tris += [a, b, c, a, c, d]
    VY, VX = np.nonzero(used)
    verts = [list(v) for v in np.stack([xs[VX], ys[VY]], 1)]
    for cell in shapely.intersection(poly, boxes[part]):
        for tri in shapely.get_parts(shapely.constrained_delaunay_triangles(cell)):
            pts = np.asarray(tri.exterior.coords)[:3]
            if abs(shapely.area(tri)) < 1e-6:
                continue
            ccw = (pts[1, 0] - pts[0, 0]) * (pts[2, 1] - pts[0, 1]) - (pts[1, 1] - pts[0, 1]) * (pts[2, 0] - pts[0, 0]) > 0
            order = (0, 1, 2) if ccw else (0, 2, 1)
            n0 = len(verts)
            verts += [list(pts[o]) for o in order]
            tris += [n0, n0 + 1, n0 + 2]
    V0 = np.asarray(verts, float)
    P = np.c_[V0, np.zeros(len(V0))]

    # simulate heights and record each step's hinge
    out_steps = []
    for st in steps:
        k = 0 if st.axis == "x" else 1
        cur = P[:, k]
        mv = cur < st.coord - EPS if st.move == "lt" else cur > st.coord + EPS
        if not mv.any():
            out_steps.append({"axis": st.axis, "c": st.coord, "hc": 0.0, "dir": 1, "mask": "", "text": st.text})
            continue
        R = P[mv].copy()
        R[:, k] = 2 * st.coord - R[:, k]
        rx0, ry0 = R[:, 0].min() - EPS, R[:, 1].min() - EPS
        rx1, ry1 = R[:, 0].max() + EPS, R[:, 1].max() + EPS
        stay = ~mv
        under = stay & (P[:, 0] >= rx0) & (P[:, 0] <= rx1) & (P[:, 1] >= ry0) & (P[:, 1] <= ry1)
        base = (P[under, 2].max() if under.any() else -GAP_CM) + GAP_CM
        top = P[mv, 2].max()
        hc = (base + top) / 2
        P[mv, k] = 2 * st.coord - P[mv, k]
        P[mv, 2] = 2 * hc - P[mv, 2]
        out_steps.append(
            {
                "axis": st.axis,
                "c": round(float(st.coord), 4),
                "hc": round(float(hc), 4),
                "dir": -1 if st.move == "lt" else 1,
                "mask": "".join("1" if m else "0" for m in mv),
                "text": st.text,
            }
        )
    return {
        "verts": np.round(V0, 3).ravel().tolist(),
        "tris": [int(t) for t in tris],
        "steps": out_steps,
        "final_h": round(float(P[:, 2].max() + GAP_CM), 2),
    }


@functools.cache
def _js() -> str:
    return (STATIC / "three.min.js").read_text() + "\n" + (STATIC / "OrbitControls.js").read_text()


def _shade(hex_colour: str, f: float) -> str:
    r, g, b = (int(hex_colour[i : i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{min(255, int(v * f)):02x}" for v in (r, g, b))


def html(g: Garment, steps: list[FoldStep], *, colour: str = "#d9a75f", height: int = 480, title: str = "", done: str = "", autoplay: bool = True) -> str:
    """Self-contained page. ``done`` is the caption after the last fold (e.g. the packet size)."""
    data = build(g, steps)
    data["title"] = title or g.label
    data["colour"], data["back"], data["autoplay"] = colour, _shade(colour, 0.78), autoplay
    data["done"] = done or "Done."
    page = _TEMPLATE.replace("__H__", str(height)).replace("/*DATA*/", json.dumps(data))
    return page.replace("/*THREE*/", _js())


_TEMPLATE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
html,body{margin:0;padding:0;font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:#faf8f4}
#wrap{position:relative;width:100%;height:__H__px}
#cv{width:100%;height:100%;display:block;border-radius:10px}
#bar{position:absolute;left:10px;right:10px;bottom:10px;display:flex;gap:8px;align-items:center;
 background:rgba(255,255,255,.92);border-radius:8px;padding:6px 10px;box-shadow:0 1px 4px rgba(0,0,0,.15)}
button{border:1px solid #c9bfae;background:#fff;border-radius:6px;padding:3px 9px;cursor:pointer;font-size:14px}
button:hover{background:#f3ede3}
#txt{flex:1;font-size:13px;color:#3b3226;line-height:1.3}
#cnt{font-size:12px;color:#7a6c58;white-space:nowrap}
#ttl{position:absolute;left:12px;top:8px;font-size:13px;color:#5a4630;background:rgba(255,255,255,.8);padding:2px 8px;border-radius:6px}
</style></head><body><div id="wrap"><canvas id="cv"></canvas><div id="ttl"></div>
<div id="bar"><button id="b0" title="restart">&#9198;</button><button id="bp" title="previous">&#9664;</button>
<button id="bplay" title="play / pause">&#9654;</button><button id="bn" title="next">&#9654;&#9654;</button>
<div id="txt"></div><div id="cnt"></div></div></div>
<script>/*THREE*/</script>
<script>
const D = /*DATA*/;
const N = D.verts.length/2, S = D.steps.length;
document.getElementById('ttl').textContent = D.title;
// keyframes: K[k] = positions (x, y, z garment coords) before step k
const K = [new Float32Array(N*3)];
for (let i=0;i<N;i++){K[0][3*i]=D.verts[2*i];K[0][3*i+1]=D.verts[2*i+1];K[0][3*i+2]=0;}
for (const st of D.steps){
  const p = K[K.length-1], q = new Float32Array(p); const a = st.axis==='x'?0:1;
  for (let i=0;i<N;i++) if (st.mask[i]==='1'){ q[3*i+a]=2*st.c-p[3*i+a]; q[3*i+2]=2*st.hc-p[3*i+2]; }
  K.push(q);
}
const cv = document.getElementById('cv');
const R = new THREE.WebGLRenderer({canvas:cv, antialias:true}); R.setPixelRatio(window.devicePixelRatio);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0xfaf8f4);
const cam = new THREE.PerspectiveCamera(40, 1, 0.5, 5000);
const ctl = new THREE.OrbitControls(cam, cv); ctl.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x9a8a70, 0.85));
const sun = new THREE.DirectionalLight(0xffffff, 0.55); sun.position.set(-60,160,120); scene.add(sun);
const geo = new THREE.BufferGeometry();
const pos = new Float32Array(N*3); geo.setAttribute('position', new THREE.BufferAttribute(pos,3));
geo.setIndex(D.tris);
const front = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({color:D.colour, side:THREE.FrontSide, roughness:.9}));
const backm = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({color:D.back, side:THREE.BackSide, roughness:.9}));
scene.add(front); scene.add(backm);
const edges = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({color:0xd03030}));
scene.add(edges);
// table
const table = new THREE.Mesh(new THREE.PlaneGeometry(4000,4000), new THREE.MeshStandardMaterial({color:0xefe8dc, roughness:1}));
table.rotation.x = -Math.PI/2; table.position.y = -0.2; scene.add(table);
// garment (x, y, z) -> three (x, z, -y)
function setPos(src){ for(let i=0;i<N;i++){pos[3*i]=src[3*i];pos[3*i+1]=src[3*i+2];pos[3*i+2]=-src[3*i+1];}
  geo.attributes.position.needsUpdate = true; geo.computeVertexNormals(); geo.computeBoundingBox(); }
function frame(k, t){ // positions during step k at fraction t
  const p = K[k], st = D.steps[k], out = new Float32Array(p); if(!st) return out;
  const a = st.axis==='x'?0:1, phi = st.dir*Math.PI*t, c=Math.cos(phi), s=Math.sin(phi);
  for (let i=0;i<N;i++) if (st.mask[i]==='1'){ const u=p[3*i+a]-st.c, w=p[3*i+2]-st.hc;
    out[3*i+a]=st.c+u*c-w*s; out[3*i+2]=st.hc+u*s+w*c; }
  return out;
}
function crease(k){ const st = D.steps[k]; if(!st){edges.visible=false;return;} edges.visible=true;
  const p=K[k]; let lo=1e9,hi=-1e9; const b=st.axis==='x'?1:0;
  for(let i=0;i<N;i++){lo=Math.min(lo,p[3*i+b]);hi=Math.max(hi,p[3*i+b]);}
  let z=0; for(let i=0;i<N;i++) z=Math.max(z,p[3*i+2]); z+=0.25; let v;
  if(st.axis==='x') v=[st.c,z,-lo, st.c,z,-hi]; else v=[lo,z,-st.c, hi,z,-st.c];
  edges.geometry.setAttribute('position', new THREE.Float32BufferAttribute(v,3)); }
function bbox(arr){let a=[1e9,1e9,1e9],b=[-1e9,-1e9,-1e9];
  for(let i=0;i<N;i++)for(let d=0;d<3;d++){a[d]=Math.min(a[d],arr[3*i+d]);b[d]=Math.max(b[d],arr[3*i+d]);} return [a,b];}
let k=0, t=0, playing=D.autoplay, pauseUntil=performance.now()+700;
const goal = {c:new THREE.Vector3(), r:60};
function aim(fast){ const [a,b]=bbox(K[Math.min(k+1,S)]); const [a0,b0]=bbox(K[k]);
  const lo=[Math.min(a[0],a0[0]),Math.min(a[1],a0[1])], hi=[Math.max(b[0],b0[0]),Math.max(b[1],b0[1])];
  goal.c.set((lo[0]+hi[0])/2, 0, -(lo[1]+hi[1])/2); goal.r = Math.max(hi[0]-lo[0], (hi[1]-lo[1])*1.15)*0.62+10;
  if(fast){ ctl.target.copy(goal.c); cam.position.set(goal.c.x, goal.r*2.1, goal.c.z+goal.r*1.25); } }
function label(){ const st=D.steps[k];
  document.getElementById('txt').textContent = k<S ? (st.text||'Fold.') : D.done;
  document.getElementById('cnt').textContent = (k<S? 'step '+(k+1)+' / '+S : S+' / '+S);
  document.getElementById('bplay').innerHTML = playing ? '&#10074;&#10074;' : '&#9654;'; }
function go(n){ k=Math.max(0,Math.min(S,n)); t=0; setPos(K[k]); crease(k); aim(false); label(); }
document.getElementById('b0').onclick=()=>{playing=false;go(0);};
document.getElementById('bp').onclick=()=>{playing=false;go(k-1);};
document.getElementById('bn').onclick=()=>{playing=false;go(k+1);};
document.getElementById('bplay').onclick=()=>{ if(k>=S) go(0); playing=!playing; pauseUntil=performance.now()+300; label(); };
function resize(){ const w=cv.clientWidth,h=cv.clientHeight; R.setSize(w,h,false); cam.aspect=w/h; cam.updateProjectionMatrix(); }
window.addEventListener('resize', resize); resize(); go(0); aim(true);
const ease = x => x<.5 ? 2*x*x : 1-Math.pow(-2*x+2,2)/2;
let last = performance.now();
function loop(now){ requestAnimationFrame(loop); const dt=Math.min(0.05,(now-last)/1000); last=now;
  if(playing && k<S && now>pauseUntil){ t += dt/1.1; if(t>=1){ k++; t=0; setPos(K[k]); crease(k); aim(false); label(); pauseUntil=now+450; if(k>=S){playing=false;label();} }
    else setPos(frame(k, ease(t))); }
  ctl.target.lerp(goal.c, 0.08);
  const want = new THREE.Vector3(goal.c.x, goal.r*2.1, goal.c.z+goal.r*1.25);
  if(!ctl._dragging) cam.position.lerp(want, 0.06);
  ctl.update(); R.render(scene, cam); }
cv.addEventListener('pointerdown',()=>{ctl._dragging=true;}); cv.addEventListener('wheel',()=>{ctl._dragging=true;});
requestAnimationFrame(loop);
</script></body></html>"""
