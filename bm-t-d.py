# -*- coding: utf-8 -*-
# build_bm_pbc_phase_thickness_clean.py
# Brick–mortar PBC cell with optional two-phase thicknesses.
# Includes "pure triangle" conversion when Nxm=Nym=0 using mirrored + midline/edge diameters.

from abaqus import *
from abaqusConstants import *
import mesh, math
import time

# --- start timing ---
t0 = time.time()
# -------------------- USER PARAMETERS --------------------
L = 1.0
Nxb, Nxm = 0, 4      # brick, mortar columns (even)
Nyb, Nym = 0, 4     # brick, mortar rows    (even)
b1 = Nyb // 2
b2 = Nyb - b1
NE_BRICK_INNER = 2    # inside each brick's inner box
NE_OTHER       = 10   # all other edges

# --- Damage / failure parameters ---
eps_fail = 0.002      # ductile failure strain (equivalent plastic strain at damage initiation)
u_fail   = eps_fail*L/NE_OTHER    # characteristic displacement at failure (softening measure)

# --- Explicit step time & loading ---

V_2     = 10  # prescribed vertical velocity at TOP boundary (length/s)
T_total = 2e-3 * ((Nyb + Nym) * 3*L) /V_2    # total analysis time (seconds) for Explicit step  =   failure strain * length of unit cell / Velocity





# Material (MPa)
E_s, nu_s, S_ys, ps = 200000.0, 0.33, 200.0, 7.8e-9  # steel-like

# Relative density target (used if solving thicknesses)
rho_target = 0.10

# === Thickness control ===
use_phase_thickness = True
tb_user = None      # e.g. 0.08
tm_user = None      # e.g. 0.05
t_ratio = 0.33       # tb / tm if solving from rho_target

# Loading (macro target strain)
epsY_target = (1.0/rho_target) * 5.0 * (S_ys/E_s)

# Names
model_name = 'B%d%dM%d%d' % (Nxb, Nyb, Nxm, Nym)
part_name  = 'BM_UnitCell'
inst_name  = 'UC_PBC'
step_name  = 'StaticLinear'

# Meshing
#seed_size = L/float(ne)

# Precision
ROUND = 9
ROUND_EDGE = 9
ROUND_NODE = 9
TOL_GEOM = 1e-9

# -------------------- DERIVED GEOMETRY --------------------
r3 = math.sqrt(3.0)
dx, dy = r3*L, 3.0*L
nX = 2*(Nxb + Nxm)
nY = (Nyb + Nym)
YMIN = 0.0
YMAX = (Nyb + Nym)*dy

# Relative-density mix coefficients
fb    = (float(Nxb)/(Nxb+max(1,Nxm))) * (float(Nyb)/(Nyb+max(1,Nym)))
C_hex = 2.0 / r3              # ~1.1547
C_tri = 2.0 * r3              # ~3.4641
eta_x = 1.0 - 1.0/(2.0*max(1, Nxb))
eta_y = 1.0 - 1.0/(1.0*max(1, Nyb))
w_eff = ((float(Nxb)/(max(1,Nxb+Nxm))) * (float(Nyb)/(max(1,Nyb+Nym)))) * eta_x * eta_y
C_coef_legacy = C_hex + w_eff*(C_tri - C_hex)

# -------------------- SKETCH HELPERS --------------------
def add(p, q): return (p[0]+q[0], p[1]+q[1])

def edge_key(p1, p2):
    a = (round(p1[0], ROUND_EDGE), round(p1[1], ROUND_EDGE))
    b = (round(p2[0], ROUND_EDGE), round(p2[1], ROUND_EDGE))
    return tuple(sorted([a, b]))

drawn = set()        # edge keys drawn
phase_map = {}       # edge key -> 'brick' or 'mortar'
geom_map = {}        # edge key -> (p1, p2) unrounded

def line_unique_tag(sk, p1, p2, phase):
    if abs(p1[0]-p2[0]) < TOL_GEOM and abs(p1[1]-p2[1]) < TOL_GEOM:
        return
    k = edge_key(p1, p2)
    if k in drawn:
        return
    sk.Line(point1=p1, point2=p2)
    drawn.add(k)
    phase_map[k] = phase
    geom_map[k]  = (p1, p2)

# 3 directions (diameters) used throughout
USET = (( r3/2.0,  0.5),  # 30°
        ( 0.0,     1.0),  # 90°
        (-r3/2.0,  0.5))  # 150°

def draw_diams_full(sk, C, side, phase):
    cx, cy = C
    for ux, uy in USET:
        p1 = (cx - side*ux, cy - side*uy)
        p2 = (cx + side*ux, cy + side*uy)
        line_unique_tag(sk, p1, p2, phase)

def draw_diams_half_up(sk, C, side, phase):
    cx, cy = C
    for ux, uy in USET:
        line_unique_tag(sk, (cx, cy), (cx + side*ux, cy + side*uy), phase)

def draw_diams_half_down(sk, C, side, phase):
    cx, cy = C
    for ux, uy in USET:
        line_unique_tag(sk, (cx - side*ux, cy - side*uy), (cx, cy), phase)

# -------------------- BUILD MODEL + SKETCH --------------------
if model_name not in mdb.models:
    mdb.Model(name=model_name)
mdl = mdb.models[model_name]

sheet = max(100.0, 1.2*(nX*dx + nY*dy))
sk = mdl.ConstrainedSketch(name='Hex_Tess_Sketch_All', sheetSize=sheet)

# Tessellation backbone (mortar phase)
base_pts = {
    'L0_1': (0.5*r3*L, 0.0),
    'L0_2': (0.5*r3*L, 0.5*L),
    'botL': (0.0,      L),
    'botR': (r3*L,     L),
    'topL': (0.0,      2.0*L),
    'topR': (r3*L,     2.0*L),
    'L1_1': (0.5*r3*L, 2.5*L),
    'L1_2': (0.5*r3*L, 3.0*L),
}
base_edges = [
    ('L0_1','L0_2'),
    ('L0_2','botL'),
    ('L0_2','botR'),
    ('botL','topL'),
    ('botR','topR'),
    ('topR','L1_1'),
    ('topL','L1_1'),
    ('L1_1','L1_2'),
]
for i in range(nX):
    for j in range(nY):
        off = (i*dx, j*dy)
        pts = {nm: add(p, off) for nm, p in base_pts.items()}
        for a_nm, b_nm in base_edges:
            line_unique_tag(sk, pts[a_nm], pts[b_nm], phase='mortar')

# Brick centers (A,B) as triangles
pointA = ( (Nxm + 1) * dx * 0.5, (Nym + 1) * dy * 0.5 )
for ix in range(Nxb):
    for iy in range(Nyb):
        draw_diams_full(sk, (pointA[0] + ix*dx, pointA[1] + iy*dy), L, phase='brick')

repBx = max(0, Nxb - 1)
repBy = max(0, Nyb - 1)
pointB = ( (Nxm + 2) * dx * 0.5, (Nym + 2) * dy * 0.5 )
for ix in range(repBx):
    for iy in range(repBy):
        draw_diams_full(sk, (pointB[0] + ix*dx, pointB[1] + iy*dy), L, phase='brick')

# Mortar centers (C,D,E,F)
pointC = ( (3*Nxm + 2*Nxb + 1) * dx * 0.5, dy * 0.5 )
for ix in range(Nxb):
    for iy in range(max(0, Nyb - b1)):
        draw_diams_full(sk, (pointC[0] + ix*dx, pointC[1] + iy*dy), L, phase='mortar')

pointD = ( (3*Nxm + 2*Nxb + 2) * dx * 0.5, 0.0 )
for ix in range(max(0, Nxb - 1)):
    for iy in range(max(0, Nyb - b1)):
        Cc = (pointD[0] + ix*dx, pointD[1] + iy*dy)
        if abs(Cc[1] - 0.0) < 1e-12: draw_diams_half_up(sk, Cc, L, phase='mortar')
        else:                        draw_diams_full(sk, Cc, L, phase='mortar')

pointE = ( (3*Nxm + 2*Nxb + 1) * dx * 0.5, dy * 0.5 + (b1 + Nym) * dy )
for ix in range(Nxb):
    for iy in range(max(0, Nyb - b2)):
        draw_diams_full(sk, (pointE[0] + ix*dx, pointE[1] + iy*dy), L, phase='mortar')

pointF = ( (3*Nxm + 2*Nxb + 1) * dx * 0.5 + dx * 0.5,
           dy * 0.5 + (b1 + Nym) * dy + dy * 0.5 )
for ix in range(max(0, Nxb - 1)):
    for iy in range(max(0, Nyb - b1)):
        Cc = (pointF[0] + ix*dx, pointF[1] + iy*dy)
        if abs(Cc[1] - YMAX) < 1e-12: draw_diams_half_down(sk, Cc, L, phase='mortar')
        else:                         draw_diams_full(sk, Cc, L, phase='mortar')

# -------------------- PURE-TRIANGLE CONVERSION (fast, clean) --------------------
def compute_bounds_from_geom():
    xs, ys = [], []
    for (p1, p2) in geom_map.values():
        xs.extend((p1[0], p2[0]))
        ys.extend((p1[1], p2[1]))
    if xs and ys:
        return min(xs), max(xs), min(ys), max(ys)
    return 0.0, nX*dx, 0.0, nY*dy

def clip_segment_y(p1, p2, y_min, y_max):
    x1,y1 = p1; x2,y2 = p2
    dyv, dxv = (y2-y1), (x2-x1)
    t0, t1 = 0.0, 1.0
    def t_at(yb):
        if abs(dyv) < 1e-16: return None
        return (yb - y1)/dyv
    # y >= y_min
    if y1 < y_min or y2 < y_min:
        t = t_at(y_min)
        if t is None: return None
        if y1 < y_min and y2 < y_min: return None
        if y1 < y_min: t0 = max(t0, t)
        if y2 < y_min: t1 = min(t1, t)
    # y <= y_max
    if y1 > y_max or y2 > y_max:
        t = t_at(y_max)
        if t is None: return None
        if y1 > y_max and y2 > y_max: return None
        if y1 > y_max: t0 = max(t0, t)
        if y2 > y_max: t1 = min(t1, t)
    if t0 > t1: return None
    q1 = (x1 + (x2-x1)*t0, y1 + (y2-y1)*t0)
    q2 = (x1 + (x2-x1)*t1, y1 + (y2-y1)*t1)
    if abs(q1[0]-q2[0])<1e-12 and abs(q1[1]-q2[1])<1e-12: return None
    return q1, q2

def clip_segment_xmin(p1, p2, xmin):
    x1,y1 = p1; x2,y2 = p2; dxv = x2-x1
    if x1 >= xmin and x2 >= xmin: return (p1,p2)
    if x1 < xmin and x2 < xmin:   return None
    if abs(dxv) < 1e-16:          return None
    t = (xmin - x1)/dxv; xm = xmin; ym = y1 + (y2-y1)*t
    if x1 < xmin: return ((xm,ym), (x2,y2))
    else:         return ((x1,y1), (xm,ym))

def clip_segment_xmax(p1, p2, xmax):
    x1,y1 = p1; x2,y2 = p2; dxv = x2-x1
    if x1 <= xmax and x2 <= xmax: return (p1,p2)
    if x1 > xmax and x2 > xmax:   return None
    if abs(dxv) < 1e-16:          return None
    t = (xmax - x1)/dxv; xm = xmax; ym = y1 + (y2-y1)*t
    if x1 > xmax: return ((xm,ym), (x2,y2))
    else:         return ((x1,y1), (xm,ym))

def clip_segment_box(p1, p2, xmin, xmax, ymin, ymax):
    seg = clip_segment_y(p1, p2, ymin, ymax)
    if seg is None: return None
    q1, q2 = seg
    seg = clip_segment_xmin(q1, q2, xmin)
    if seg is None: return None
    q1, q2 = seg
    seg = clip_segment_xmax(q1, q2, xmax)
    if seg is None: return None
    q1, q2 = seg
    if abs(q1[0]-q2[0])<1e-12 and abs(q1[1]-q2[1])<1e-12: return None
    return q1, q2

def draw_diameters_clipped_box(sk, C, L, box, phase='brick'):
    (xmin, xmax, ymin, ymax) = box
    for ux, uy in USET:
        p1 = (C[0] - L*ux, C[1] - L*uy)
        p2 = (C[0] + L*ux, C[1] + L*uy)
        seg = clip_segment_box(p1, p2, xmin, xmax, ymin, ymax)
        if seg is not None:
            q1, q2 = seg
            line_unique_tag(sk, q1, q2, phase)

if (Nxm == 0) and (Nym == 0):
    # Bounds from current sketch
    XMIN, XMAX, YMIN_b, YMAX_b = compute_bounds_from_geom()
    W = XMAX - XMIN; H = YMAX_b - YMIN_b
    XMID = XMIN + 0.5*W
    box  = (XMIN, XMAX, YMIN_b, YMAX_b)

    # 1) Mirror all edges about x = W/2 (no Abaqus feature needed)
    for (p1, p2) in list(geom_map.values()):  # snapshot
        p1m = (2.0*XMID - p1[0], p1[1])
        p2m = (2.0*XMID - p2[0], p2[1])
        # keep original phase if present, else mortar
        k = edge_key(p1, p2); phase = phase_map.get(k, 'mortar')
        line_unique_tag(sk, p1m, p2m, phase)

    # 2) Hex center grid along x=W/2, 0 and W
    Xc_list = [XMIN + 0.5*r3*L + i*dx for i in range(max(0, nX))]
    Yc_list = [YMIN_b + 1.5*L     + j*dy for j in range(max(0, nY))]
    # choose midline xc ≈ XMID
    TOLX = max(1e-9, 1e-9*W)
    Xc_mid = next((xc for xc in Xc_list if abs(xc - XMID) <= TOLX), XMID)

    # 3) Midline x=W/2: shift +1.5L, clip to 0..H; extra pass at y=0 (clip y>=0)
    for yc in Yc_list:
        draw_diameters_clipped_box(sk, (Xc_mid, yc + 1.5*L), L, box, phase='brick')
    draw_diameters_clipped_box(sk, (Xc_mid, YMIN_b), L, box, phase='brick')

    # 4) Left x=0 and Right x=W: same shift and extra y=0 pass
    Xc_left, Xc_right = XMIN, XMAX
    for yc in Yc_list:
        draw_diameters_clipped_box(sk, (Xc_left,  yc + 1.5*L), L, box, phase='brick')
        draw_diameters_clipped_box(sk, (Xc_right, yc + 1.5*L), L, box, phase='brick')
    draw_diameters_clipped_box(sk, (Xc_left,  YMIN_b), L, box, phase='brick')
    draw_diameters_clipped_box(sk, (Xc_right, YMIN_b), L, box, phase='brick')

# -------------------- PART --------------------
if part_name in mdl.parts: del mdl.parts[part_name]
prt = mdl.Part(name=part_name, dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
prt.BaseWire(sketch=sk)

# -------------------- MATERIAL/SECTIONS --------------------
mat_name = 'Mat_200GPa_200MPa'
if mat_name in mdl.materials: del mdl.materials[mat_name]
mat = mdl.Material(name=mat_name)
mat.Elastic(table=((E_s, nu_s),))
mat.Plastic(table=((S_ys, 0.0),))  # perfectly plastic
mat.Density(table=((ps, ), ))

# --- Ductile damage: initiation at failure strain, evolution by displacement (explicit-friendly) ---
#   - Initiation: (equiv plastic strain at failure, stress triaxiality, strain rate)
#     use 0 for triaxiality & rate if you don't have dependency data.
#mat.DuctileDamageInitiation(table=((eps_fail, 0.0, 0.0),))

#   - Evolution: displacement-based softening to deletion
#     (linear softening; Abaqus/Explicit deletes elements when damage = 1)
#mat.DamageEvolution(type=DISPLACEMENT, table=((u_fail,),), softening=LINEAR)


mat.DuctileDamageInitiation(table=((eps_fail, 0.0, 0.0), ))
mat.ductileDamageInitiation.DamageEvolution(type=DISPLACEMENT, table=((u_fail, ), ))



# beam sections (two-phase thickness kept as before)
prof_brick = 'Rect_a1_b_tb'
prof_mort  = 'Rect_a1_b_tm'
for pn in (prof_brick, prof_mort):
    if pn in mdl.profiles: del mdl.profiles[pn]
mdl.RectangularProfile(name=prof_brick, a=1.0, b=tb)
mdl.RectangularProfile(name=prof_mort,  a=1.0, b=tm)

sec_brick = 'BeamSect_BRICK'
sec_mort  = 'BeamSect_MORTAR'
for sn in (sec_brick, sec_mort):
    if sn in mdl.sections: del mdl.sections[sn]
mdl.BeamSection(name=sec_brick, integration=DURING_ANALYSIS, profile=prof_brick, material=mat_name)
mdl.BeamSection(name=sec_mort,  integration=DURING_ANALYSIS, profile=prof_mort,  material=mat_name)


# -------------------- CONNECTIVITY-BASED PHASE MAPPING (fast) --------------------
def vxy_from_vertex(v):
    try: vv = prt.vertices[v] if isinstance(v, (int, long)) else v
    except NameError: vv = prt.vertices[v] if isinstance(v, int) else v
    try:    p = vv.pointOn[0]
    except: p = getattr(vv, 'coordinates', vv.pointOn)
    return (p[0], p[1])

def vkey_from_vertex(v):
    x, y = vxy_from_vertex(v)
    return (round(x, ROUND_NODE), round(y, ROUND_NODE))

# Build degrees, classify edges
deg = {}
edge_idx_is_brick = []
edge_idx_is_mort  = []
for e in prt.edges:
    v1k, v2k = vkey_from_vertex(e.getVertices()[0]), vkey_from_vertex(e.getVertices()[1])
    deg[v1k] = deg.get(v1k, 0) + 1
    deg[v2k] = deg.get(v2k, 0) + 1
# second pass to classify
BRICK_DEGS = {5, 6}
for e in prt.edges:
    v1k, v2k = vkey_from_vertex(e.getVertices()[0]), vkey_from_vertex(e.getVertices()[1])
    if (deg.get(v1k,0) in BRICK_DEGS) or (deg.get(v2k,0) in BRICK_DEGS):
        edge_idx_is_brick.append(e.index)
    else:
        edge_idx_is_mort.append(e.index)

def make_set_from_indices(part, name, idx_list):
    """
    Build a Set(name=...) from a list of geometry edge indices.
    Speeds up by:
      1) sorting + merging into contiguous runs,
      2) creating one temporary set per run using a slice,
      3) unioning runs into the final set.
    """
    if name in part.sets:
        del part.sets[name]
    if not idx_list:
        return False

    # 1) sort and build contiguous runs [i0..i1]
    idx_sorted = sorted(set(int(i) for i in idx_list))
    runs = []
    i0 = i1 = idx_sorted[0]
    for k in idx_sorted[1:]:
        if k == i1 + 1:
            i1 = k
        else:
            runs.append((i0, i1))
            i0 = i1 = k
    runs.append((i0, i1))

    # 2) create a base set from the first run
    r0s, r0e = runs[0]
    base_name = name  # final set's name
    part.Set(name=base_name, edges=part.edges[r0s:r0e+1])

    # 3) union in the remaining runs
    for (rs, re) in runs[1:]:
        tmp = '%s_tmp_%d_%d' % (name, rs, re)
        if tmp in part.sets:
            del part.sets[tmp]
        part.Set(name=tmp, edges=part.edges[rs:re+1])
        part.SetByBoolean(name=base_name,
                          sets=(part.sets[base_name], part.sets[tmp]),
                          operation=UNION)
        del part.sets[tmp]

    return True



made_brick  = make_set_from_indices(prt, 'Edges_BRICK',  sorted(set(edge_idx_is_brick)))
made_mortar = make_set_from_indices(prt, 'Edges_MORTAR', sorted(set(edge_idx_is_mort)))

# Assign sections
if made_brick:
    prt.SectionAssignment(region=prt.sets['Edges_BRICK'],  sectionName=sec_brick)
    prt.assignBeamSectionOrientation(region=prt.sets['Edges_BRICK'],  method=N1_COSINES, n1=(0.0, 0.0, 1.0))
if made_mortar:
    prt.SectionAssignment(region=prt.sets['Edges_MORTAR'], sectionName=sec_mort)
    prt.assignBeamSectionOrientation(region=prt.sets['Edges_MORTAR'], method=N1_COSINES, n1=(0.0, 0.0, 1.0))

# -------------------- STEP (Dynamic Explicit) --------------------
if step_name in mdl.steps: del mdl.steps[step_name]
mdl.ExplicitDynamicsStep(name=step_name, previous='Initial',
                         timePeriod=T_total, improvedDtMethod=ON)

# -------------------- OUTPUTS --------------------
# Field output: add RF, keep 200 intervals
if 'FOUT_MAIN' in mdl.fieldOutputRequests: del mdl.fieldOutputRequests['FOUT_MAIN']
mdl.FieldOutputRequest(name='FOUT_MAIN', createStepName=step_name,
                       variables=('S','MISES','PEEQ','U','RF'),
                       numIntervals=200)

# History outputs: ONLY RF2 at TOP_ALL and U2 at T_0000, 200 intervals
for nm in list(mdl.historyOutputRequests.keys()):
    del mdl.historyOutputRequests[nm]

mdl.HistoryOutputRequest(name='HO_RF2_TOP', createStepName=step_name,
                         variables=('RF2',), region=a.sets['TOP_ALL'],
                         numIntervals=200)

mdl.HistoryOutputRequest(name='HO_U2_T0', createStepName=step_name,
                         variables=('U2',), region=a.sets['T_0000'],
                         numIntervals=200)


# -------------------- MESH (phase-aware, per-brick inner boxes) --------------------
# Params for element counts


# Shrink margins for each brick component's inner box
X_SHRINK = math.sqrt(3.0) * L   # in x (each side)
Y_SHRINK =  L              # in y (each side)

# Helpers to read geometry coords
def _vxy_from_vertex(v):
    try:
        vv = prt.vertices[v] if isinstance(v, (int, long)) else v
    except NameError:
        vv = prt.vertices[v] if isinstance(v, int) else v
    try:
        p = vv.pointOn[0]
    except Exception:
        p = getattr(vv, 'coordinates', vv.pointOn)
    return (p[0], p[1])

def _edge_endpoints_xy(e):
    v1, v2 = e.getVertices()
    return _vxy_from_vertex(v1), _vxy_from_vertex(v2)

def _edge_midpoint_xy(e):
    (x1,y1), (x2,y2) = _edge_endpoints_xy(e)
    return (0.5*(x1+x2), 0.5*(y1+y2))

# Build a fast lookup for brick edges
if 'Edges_BRICK' in prt.sets:
    brick_ea  = prt.sets['Edges_BRICK'].edges
else:
    brick_ea  = prt.edges[0:0]  # empty
if 'Edges_MORTAR' in prt.sets:
    mortar_ea = prt.sets['Edges_MORTAR'].edges
else:
    mortar_ea = prt.edges[0:0]

# ---- Find connected components of the BRICK geometry graph ----
# Build vertex->edge adjacency for brick edges
from collections import defaultdict, deque

def _vkey(x, y, r=ROUND_NODE):
    return (round(x, r), round(y, r))

v2e = defaultdict(list)    # vertex_key -> list of edge indices
brick_idx = [int(e.index) for e in brick_ea]
for e in brick_ea:
    (x1,y1), (x2,y2) = _edge_endpoints_xy(e)
    v2e[_vkey(x1,y1)].append(int(e.index))
    v2e[_vkey(x2,y2)].append(int(e.index))

# Build edge->vertex keys for fast traversal
e2v = {}
for e in brick_ea:
    (x1,y1), (x2,y2) = _edge_endpoints_xy(e)
    e2v[int(e.index)] = (_vkey(x1,y1), _vkey(x2,y2))

# BFS over edges via shared vertices to get components
unseen = set(brick_idx)
components = []  # list of lists of edge indices
while unseen:
    seed = unseen.pop()
    comp = [seed]
    q = deque([seed])
    while q:
        ei = q.popleft()
        for vk in e2v[ei]:
            for ej in v2e[vk]:
                if ej in unseen:
                    unseen.remove(ej)
                    comp.append(ej)
                    q.append(ej)
    components.append(sorted(comp))

# ---- For each component, compute bbox, shrink, and collect "inner" brick edges ----
def _bbox_of_edges(edge_indices):
    xs, ys = [], []
    for i in edge_indices:
        e = prt.edges[i]
        (x1,y1), (x2,y2) = _edge_endpoints_xy(e)
        xs.extend((x1,x2)); ys.extend((y1,y2))
    return (min(xs), max(xs), min(ys), max(ys)) if xs else (0,0,0,0)

def _inside_box(pt, box):
    x,y = pt; xmin,xmax,ymin,ymax = box
    return (xmin <= x <= xmax) and (ymin <= y <= ymax)

brick_inner_idx = set()  # brick edges inside their own inner boxes
for comp in components:
    xmin, xmax, ymin, ymax = _bbox_of_edges(comp)
    # shrink per your spec on EACH side
    xmin_in = xmin + X_SHRINK
    xmax_in = xmax - X_SHRINK
    ymin_in = ymin + Y_SHRINK
    ymax_in = ymax - Y_SHRINK
    # guard: if shrink inverts the box, skip inner selection for this comp
    if xmin_in >= xmax_in or ymin_in >= ymax_in:
        continue
    box_in = (xmin_in, xmax_in, ymin_in, ymax_in)
    for i in comp:
        e = prt.edges[i]
        if _inside_box(_edge_midpoint_xy(e), box_in):
            brick_inner_idx.add(i)

# ---- Build EdgeArrays for seeding ----
def _edgearray_from_indices(idx_list):
    if not idx_list:
        return prt.edges[0:0]
    idxs = sorted(set(int(i) for i in idx_list))
    # group contiguous runs
    runs = []
    s = e0 = idxs[0]
    for k in idxs[1:]:
        if k == e0 + 1:
            e0 = k
        else:
            runs.append((s, e0))
            s = e0 = k
    runs.append((s, e0))
    # concatenate slices
    ea = prt.edges[0:0]
    first = True
    for a,b in runs:
        if first:
            ea = prt.edges[a:b+1]; first = False
        else:
            tmp = prt.edges[a:b+1]
            ea = tuple(list(ea) + list(tmp))
    return ea

# Brick edges: split into inner (coarse) and outer (fine)
brick_inner_ea = _edgearray_from_indices(brick_inner_idx)
brick_outer_idx = [i for i in brick_idx if i not in brick_inner_idx]
brick_outer_ea = _edgearray_from_indices(brick_outer_idx)

# Mortar edges: always fine
mortar_idx = [int(e.index) for e in mortar_ea]
mortar_ea_all = _edgearray_from_indices(mortar_idx)

# ---- Seed by NUMBER ----
if len(brick_inner_ea) > 0:
    prt.seedEdgeByNumber(edges=brick_inner_ea, number=NE_BRICK_INNER, constraint=FINER)
if len(brick_outer_ea) > 0:
    prt.seedEdgeByNumber(edges=brick_outer_ea, number=NE_OTHER,       constraint=FINER)
if len(mortar_ea_all) > 0:
    prt.seedEdgeByNumber(edges=mortar_ea_all, number=NE_OTHER,       constraint=FINER)

# Element type + mesh
prt.setElementType(regions=(prt.edges,), elemTypes=(mesh.ElemType(elemCode=B21),))
prt.generateMesh()

# Report
print("Brick components found: %d" % len(components))
print("Brick inner (2/bar): %d edges; Brick outer (10/bar): %d edges; Mortar (10/bar): %d edges"
      % (len(brick_inner_ea), len(brick_outer_ea), len(mortar_ea_all)))
print("Per-brick inner box shrink: dx=%.3f, dy=%.3f" % (X_SHRINK, Y_SHRINK))



# -------------------- ASSEMBLY + PBC --------------------
a = mdl.rootAssembly
a.DatumCsysByDefault(CARTESIAN)
if inst_name in a.instances: del a.instances[inst_name]
a.Instance(name=inst_name, part=prt, dependent=ON); a.regenerate()
inst = a.instances[inst_name]; nodes = inst.nodes

xs = [nd.coordinates[0] for nd in nodes]; ys = [nd.coordinates[1] for nd in nodes]
xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
span = max(xmax - xmin, ymax - ymin, 1.0)
XTOL = max(1e-6, 1e-8*span); YTOL = XTOL

left_nodes, right_nodes, bottom_nodes, top_nodes = [], [], [], []
for nd in nodes:
    x,y,_ = nd.coordinates
    if abs(x - xmin) <= XTOL: left_nodes.append(nd)
    if abs(x - xmax) <= XTOL: right_nodes.append(nd)
    if abs(y - ymin) <= YTOL: bottom_nodes.append(nd)
    if abs(y - ymax) <= YTOL: top_nodes.append(nd)
left_nodes.sort(  key=lambda n: round(n.coordinates[1], ROUND))
right_nodes.sort( key=lambda n: round(n.coordinates[1], ROUND))
bottom_nodes.sort(key=lambda n: round(n.coordinates[0], ROUND))
top_nodes.sort(   key=lambda n: round(n.coordinates[0], ROUND))

def is_corner(n, x0, y0):
    x,y,_ = n.coordinates
    return (abs(x-x0)<=XTOL) and (abs(y-y0)<=YTOL)
BL = BR = TL = TR = None
for n in nodes:
    if is_corner(n,xmin,ymin): BL = n
    if is_corner(n,xmax,ymin): BR = n
    if is_corner(n,xmin,ymax): TL = n
    if is_corner(n,xmax,ymax): TR = n
bottom_nodes = [n for n in bottom_nodes if not (n==BL or n==BR)]
top_nodes    = [n for n in top_nodes    if not (n==TL or n==TR)]

nLR = min(len(left_nodes), len(right_nodes))
nTB = min(len(top_nodes),  len(bottom_nodes))
if nLR == 0 or nTB == 0:
    raise RuntimeError('No boundary pairs found. Check mesh/tolerances.')

def nset_name(prefix, idx): return '%s_%04d' % (prefix, idx)

# Rotations equal across opposite edges
for k in range(nLR):
    Lk = left_nodes[k]; Rk = right_nodes[k]
    setL = nset_name('L', k); setR = nset_name('R', k)
    for s in (setL,setR):
        if s in a.sets: del a.sets[s]
    a.Set(name=setL, nodes=inst.nodes.sequenceFromLabels((Lk.label,)))
    a.Set(name=setR, nodes=inst.nodes.sequenceFromLabels((Rk.label,)))
    mdl.Equation(name='PBC_UR3_LR_%04d' % k, terms=(( 1.0, setR, 6), (-1.0, setL, 6)))
for j in range(nTB):
    Bj = bottom_nodes[j]; Tj = top_nodes[j]
    setB = nset_name('B', j); setT = nset_name('T', j)
    for s in (setB,setT):
        if s in a.sets: del a.sets[s]
    a.Set(name=setB, nodes=inst.nodes.sequenceFromLabels((Bj.label,)))
    a.Set(name=setT, nodes=inst.nodes.sequenceFromLabels((Tj.label,)))
    mdl.Equation(name='PBC_UR3_TB_%04d' % j, terms=(( 1.0, setT, 6), (-1.0, setB, 6)))

# Linear displacements
for k in range(nLR):
    setL = nset_name('L', k); setR = nset_name('R', k)
    mdl.Equation(name='PBC_U2_LR_%04d' % k, terms=(( 1.0, setR, 2), (-1.0, setL, 2)))
for j in range(nTB):
    setB = nset_name('B', j); setT = nset_name('T', j)
    mdl.Equation(name='PBC_U1_TB_%04d' % j, terms=(( 1.0, setT, 1), (-1.0, setB, 1)))
if nTB >= 2:
    setB0 = nset_name('B', 0); setT0 = nset_name('T', 0)
    for j in range(1, nTB):
        setB = nset_name('B', j); setT = nset_name('T', j)
        mdl.Equation(name='PBC_U2_TB_JUMP_%04d' % j,
                     terms=(( 1.0, setT, 2), (-1.0, setB, 2), (-1.0, setT0, 2), ( 1.0, setB0, 2)))
if nLR >= 2:
    setL0 = nset_name('L', 0); setR0 = nset_name('R', 0)
    for k in range(1, nLR):
        setL = nset_name('L', k); setR = nset_name('R', k)
        mdl.Equation(name='PBC_U1_LR_JUMP_%04d' % k,
                     terms=(( 1.0, setR, 1), (-1.0, setL, 1), (-1.0, setR0, 1), ( 1.0, setL0, 1)))

# -------------------- OUTPUTS & LOAD --------------------
W = xmax - xmin; H = ymax - ymin

if 'TOP_ALL' in a.sets: del a.sets['TOP_ALL']
a.Set(name='TOP_ALL', nodes=inst.nodes.sequenceFromLabels(tuple(nd.label for nd in top_nodes)))

if 'BOTTOM_ALL' in a.sets: del a.sets['BOTTOM_ALL']
a.Set(name='BOTTOM_ALL', nodes=inst.nodes.sequenceFromLabels(tuple(nd.label for nd in bottom_nodes)))



# -------------------- BOUNDARY CONDITIONS (Explicit) --------------------
# Remove any old displacement BCs that set U2 on T_0000 / B_0000
for nm in ('BC_U2_B0','BC_U2_T0','BC_Anchor'):
    if nm in mdl.boundaryConditions:
        del mdl.boundaryConditions[nm]

# Keep a minimal anchor for rigid in-plane modes (U1 & UR3 on a single bottom ref pair)
#   (U2 is controlled separately on the entire bottom edge)
mdl.DisplacementBC(name='BC_Anchor', createStepName='Initial',
                   region=a.sets['B_0000'], u1=0.0, u2=UNSET, ur3=0.0)

# Enforce U2 = 0 on ALL bottom boundary nodes (to keep damage effects off the boundary)
mdl.DisplacementBC(name='BC_Bottom_U2_Zero', createStepName='Initial',
                   region=a.sets['BOTTOM_ALL'], u2=0.0)

# Apply constant vertical velocity V_2 on ALL top boundary nodes
mdl.VelocityBC(name='BC_Top_V2', createStepName=step_name,
               region=a.sets['TOP_ALL'], v2=V_2)

# (No displacement U2 on T_0000 in the loading step)


# -------------------- JOB --------------------
# encode eps_fail into a filename-safe token (e.g., 0.20 -> 0p20)
epsf_token = ('%.3f' % eps_fail).replace('.', 'p')
job_name = 'Job_B%dx%d_M%dx%d_epsf%s' % (Nxb, Nyb, Nxm, Nym, epsf_token)
if job_name in mdb.jobs: del mdb.jobs[job_name]
mdb.Job(name=job_name, model=model_name)


# -------------------- SUMMARY --------------------
rho_eff = ((1.0 - w_eff) * C_hex * (tm / L)) + (w_eff * C_tri * (tb / L))
print('--- SUMMARY ---')
print('tb (brick)=%.6g, tm (mortar)=%.6g [tb/tm=%.4g]' % (tb, tm, (tb/tm if tm>0 else float('nan'))))
print('rho_target=%.6g, rho_eff=%.6g, fb=%.6g' % (rho_target, rho_eff, fb))
print('Spans: W=%.6g, H=%.6g' % (W,H))
#print('Applied epsY_target=%.6g -> deltaY=%.6g' % (epsY_target, deltaY))
print('Created Job:', job_name)

# --- stop timing ---
t1 = time.time()
elapsed = t1 - t0
print("Elapsed time: %.2f seconds" % elapsed)