# uncomment if running from __main__
# import init_paths
import numpy as np
import transforms3d.quaternions as txq
import logging
from thirdparty import binvox_rw
from copy import deepcopy
import matplotlib.pyplot as plt
import transforms3d.euler as txe
from import_open3d import *


# Unit = meters

def pose_matrix(pose):
  T = np.eye(4)
  T[:3, 3]  = pose['translation']
  T[:3, :3] = txq.quat2mat(pose['rotation'])
  return T


def project(oX, cTo, K, A=np.eye(3)):
  oX = np.vstack((oX.T, np.ones(len(oX))))
  cX = cTo @ oX
  cX = K @ cX[:3]
  cX /= cX[2]
  cX = A @ cX
  return cX[:2].T


def tform_points(T, X):
  """
  X: Nx3
  T: 4x4 homogeneous
  """
  X = np.vstack((X.T, np.ones(len(X))))
  X = T @ X
  X = X[:3].T
  return X


def softmax(x):
  return np.exp(x) / sum(np.exp(x))


def rotmat_from_vecs(v1, v2=np.asarray([0, 0, 1])):
  """
  Returns a rotation matrix R_1_2
  :param v1: vector in frame 1
  :param v2: vector in frame 2
  :return:
  """
  v1 = v1 / np.linalg.norm(v1)
  v2 = v2 / np.linalg.norm(v2)
  v = np.cross(v2, v1)
  vx = np.asarray([
    [0,    -v[2], +v[1], 0],
    [+v[2], 0,    -v[0], 0],
    [-v[1], +v[0], 0,    0],
    [0,     0,     0,    0]])
  dotp = np.dot(v1, v2)

  if np.abs(dotp + 1) < 1e-3:
    R = np.eye(4)
    x = np.cross(v2, [1, 0, 0])
    R[:3, :3] = txe.axangle2mat(x, np.pi)
  else:
    R = np.eye(4) + vx + np.dot(vx, vx)/(1+dotp)
  return R


def create_hand_geoms(joint_locs, joint_color, lines_color=None,
    kept_joints=None, joint_sphere_radius_mm=4.0, bone_cylinder_radius_mm=2.5):
  if lines_color is None:
    lines_color=np.asarray([224.0, 172.0, 105.0])/255
  geoms = []
  
  for idx, j in enumerate(joint_locs):
    m = o3dg.TriangleMesh.create_sphere(radius=joint_sphere_radius_mm*1e-3,
        resolution=10)
    m.compute_vertex_normals()
    if kept_joints is None:
      c = joint_color
    else:
      c = joint_color if kept_joints[idx] else [0, 0, 0]
    m.paint_uniform_color(c)
    m.translate(j)
    geoms.append(m)

  line_ids = get_hand_line_ids()
  for line_idx, (idx0, idx1) in enumerate(line_ids):
    bone = joint_locs[idx0] - joint_locs[idx1]
    h = np.linalg.norm(bone)
    l = o3dg.TriangleMesh.create_cylinder(radius=bone_cylinder_radius_mm*1e-3,
        height=h, resolution=10)
    l.paint_uniform_color(lines_color)
    l.compute_vertex_normals()
    l.translate([0, 0, -h/2.0])
    T = rotmat_from_vecs(bone, [0, 0, 1])
    T[:3, 3] = joint_locs[idx0]
    l.transform(T)
    geoms.append(l)

  return geoms


def create_hand_geoms_lines(joint_locs, joint_color=[0, 0, 0],
    lines_color=[0, 0, 0]):
  geoms = []
  
  pc = o3dg.PointCloud()
  pc.points = o3du.Vector3dVector(joint_locs)
  pc.paint_uniform_color(joint_color)
  geoms.append(pc)

  ls = o3dg.LineSet()
  ls.points = pc.points
  line_ids = get_hand_line_ids()
  ls.lines = o3du.Vector2iVector(line_ids)
  lines_color = [lines_color for _ in range(len(line_ids))]
  ls.colors = o3du.Vector3dVector(np.asarray(lines_color))
  geoms.append(ls)

  return geoms


def show_prediction(surface_colors, mesh, binvox, joint_locs,
    kept_joints=[None, None], mesh_scale=1e-3):
  if not isinstance(mesh, o3dg.TriangleMesh):
    mesh = o3dio.read_triangle_mesh(mesh)
    mesh.scale(mesh_scale, center=False)
  mesh.compute_vertex_normals()
  if not isinstance(binvox, binvox_rw.Voxels):
    with open(binvox, 'rb') as f:
      binvox = binvox_rw.read_as_coord_array(f)
  pc = binvox2pc(binvox)
  pc = create_pc(pc, surface_colors)
  # draw_geometries_with_colormap([pc], apply_sigmoid=False)
  mesh = transfer_color_pc2mesh(pc, mesh)
  geoms = [apply_colormap(mesh, apply_sigmoid=False)]
  hand_colors = [[0, 1, 0], [1, 0, 0]]
  for hand_idx, hand in joint_locs.items():
    geoms.extend(create_hand_geoms(hand, joint_color=hand_colors[hand_idx],
      kept_joints=kept_joints[hand_idx]))
  o3dv.draw_geometries(geoms)


def get_hand_line_ids():
    line_ids = []
    for finger in range(5):
        base = 4*finger + 1
        line_ids.append([0, base])
        for j in range(3):
            line_ids.append([base+j, base+j+1])
    line_ids = np.asarray(line_ids, dtype=int)
    return line_ids


def closest_linesegment_point(l0, l1, p):
  """
  p: N x 3
  l0, l1: M x 3
  out: N x M x 3
  """
  p  = np.broadcast_to(p[:, np.newaxis, :],  (len(p), len(l0), 3))
  l0 = np.broadcast_to(l0[np.newaxis, :, :], (len(p), len(l0), 3))
  l1 = np.broadcast_to(l1[np.newaxis, :, :], (len(p), len(l1), 3))
  
  llen = np.linalg.norm(l1 - l0, axis=-1, keepdims=True)
  lu = (l1 - l0) / llen

  v = p - l0
  d = np.sum(v * lu, axis=-1, keepdims=True)
  d = np.clip(d, a_min=0, a_max=llen)

  out = l0 + d * lu
  return out


def texture_proc(colors, a=0.05, invert=False):
  idx = colors > 0 
  ci = colors[idx]
  if len(ci) == 0:
      return colors
  if invert:
      ci = 1 - ci
  # fit a sigmoid
  x1 = min(ci); y1 = a 
  x2 = max(ci); y2 = 1-a 
  lna = np.log((1 - y1) / y1) 
  lnb = np.log((1 - y2) / y2) 
  k = (lnb - lna) / (x1 - x2) 
  mu = (x2*lna - x1*lnb) / (lna - lnb)
  # apply the sigmoid
  ci = np.exp(k * (ci-mu)) / (1 + np.exp(k * (ci-mu)))
  colors[idx] = ci
  return colors


def apply_colormap(geom, apply_sigmoid=True):
  if isinstance(geom, o3dg.PointCloud):
    colors = np.asarray(geom.colors)[:, 0]
  elif isinstance(geom, o3dg.TriangleMesh):
    colors = np.asarray(geom.vertex_colors)[:, 0]
  else:
    raise NotImplementedError
  if apply_sigmoid:
    colors = texture_proc(colors)
  colors = plt.cm.inferno(colors)[:, :3]
  colors = o3du.Vector3dVector(colors)
  if isinstance(geom, o3dg.PointCloud):
    geom.colors = colors
  elif isinstance(geom, o3dg.TriangleMesh):
    geom.vertex_colors = colors
  return geom


def draw_geometries_with_colormap(geoms, apply_sigmoid=True):
  draw_geoms = []
  for geom in geoms:
    geom = deepcopy(geom)
    draw_geoms.append(apply_colormap(geom, apply_sigmoid))
  o3dv.draw_geometries(draw_geoms)


def mesh2pc(mesh, n_points):
  if isinstance(mesh, str):
    mesh = o3dio.read_triangle_mesh(mesh)
  pc = mesh.sample_points_poisson_disk(n_points)
  return np.asarray(pc.points)


def binvox2pc(binvox):
  if isinstance(binvox, str):
    with open(binvox, 'rb') as f:
      binvox = binvox_rw.read_as_coord_array(f)
  points = binvox.data.T + 0.5
  points /= np.asarray(binvox.dims, dtype=float)
  points = binvox.scale * points + np.asarray(binvox.translate, dtype=float)
  points /= 1000.0
  return points


def create_pc(points, colors=None, normals=None):
  pc = o3dg.PointCloud()
  pc.points = o3du.Vector3dVector(points)
  if colors is not None:
    colors = np.tile(colors[:, np.newaxis], (1, 3))
    pc.colors = o3du.Vector3dVector(colors)
  if normals is not None:
    pc.normals = o3du.Vector3dVector(normals)
  return pc


def transfer_color_mesh2pc(mesh, pc, n_max_nbrs=5, interp_dist_thresh_mm=None):
  logger = logging.getLogger(__name__)
  tree = o3dg.KDTreeFlann(mesh)
  
  if interp_dist_thresh_mm is None:
    avg_dist, max_dist = calc_nnbr_distance_stats(create_pc(pc))
    interp_dist_thresh = max_dist * 3.0
  else:
    interp_dist_thresh = interp_dist_thresh_mm * 1e-3
  
  assert(mesh.has_vertex_colors())
  mesh_colors = np.asarray(mesh.vertex_colors)[:, 0]
  pc_colors = []
  for p in pc:
    n_nbrs, idx_nbrs, dist2_nbrs = tree.search_radius_vector_3d(
        p, interp_dist_thresh)
    if n_nbrs == 0:
      logger.warning('No neighbours for point {:f} {:f} {:f}'.format(*p))
      pc_colors.append(0)
      continue
    # choose closest nonzero neighbors
    cs = mesh_colors[idx_nbrs]
    valid_idx = np.flatnonzero(cs > 0.0)
    if len(valid_idx) == 0:
      logger.debug(
          'All neighbors have 0 color for point {:f} {:f} {:f}'.format(*p))
      pc_colors.append(0)
      continue
    else:
      valid_idx = valid_idx[:n_max_nbrs]
    cs = cs[valid_idx]
    ws = softmax(np.sqrt(dist2_nbrs)[valid_idx])
    c  = np.dot(cs, ws)
    pc_colors.append(c)
  return np.asarray(pc_colors)


def transfer_normals_mesh2pc(mesh, pc, interp_dist_thresh_mm=None):
  logger = logging.getLogger(__name__)
  tree = o3dg.KDTreeFlann(mesh)
  
  if interp_dist_thresh_mm is None:
    avg_dist, max_dist = calc_nnbr_distance_stats(create_pc(pc))
    interp_dist_thresh = max_dist * 3.0
  else:
    interp_dist_thresh = interp_dist_thresh_mm * 1e-3
  
  if not mesh.has_vertex_normals():
    mesh.compute_vertex_normals()
  mesh_normals = np.asarray(mesh.vertex_normals)
  pc_normals = []
  for p in pc:
    n_nbrs, idx_nbrs, dist2_nbrs = tree.search_hybrid_vector_3d(
        p, interp_dist_thresh, 5)
    if n_nbrs == 0:
      logger.warning('No neighbours for point {:f} {:f} {:f}'.format(*p))
      pc_normals.append(np.asarray([1, 0, 0]))
      continue
    ns = mesh_normals[idx_nbrs]
    ws = softmax(np.sqrt(dist2_nbrs))
    n  = np.sum(ns * ws[:, np.newaxis], axis=0)
    pc_normals.append(n)
  pc_normals = np.asarray(pc_normals)
  pc_normals /= np.linalg.norm(pc_normals, axis=1, keepdims=True)
  return pc_normals


def transfer_color_pc2mesh(pc, mesh, interp_dist_thresh_mm=None):
  logger = logging.getLogger(__name__)
  mesh = deepcopy(mesh)
  assert(pc.has_colors())
  pc_colors = np.asarray(pc.colors)[:, 0]
  tree = o3dg.KDTreeFlann(pc)
  
  if interp_dist_thresh_mm is None:
    avg_dist, max_dist = calc_nnbr_distance_stats(pc, tree)
    interp_dist_thresh = max_dist * 1.5
  else:
    interp_dist_thresh = interp_dist_thresh_mm * 1e-3
  
  vertex_colors = []
  for p in np.asarray(mesh.vertices):
    n_nbrs, idx_nbrs, dist2_nbrs = tree.search_hybrid_vector_3d(
        p, interp_dist_thresh, 3)
    if n_nbrs == 0:
      logger.warning('No neighbours for point {:f} {:f} {:f}'.format(*p))
      vertex_colors.append(0)
      continue
    cs = pc_colors[idx_nbrs]
    valid_idx = cs > 0.0
    if np.sum(valid_idx) == 0:
      logger.warning(
          'All neighbors have 0 color for point {:f} {:f} {:f}'.format(*p))
      vertex_colors.append(0)
      continue
    cs = cs[valid_idx]
    ws = softmax(np.sqrt(dist2_nbrs)[valid_idx])
    c  = np.dot(cs, ws)
    vertex_colors.append(c)
  vertex_colors = np.asarray(vertex_colors)
  vertex_colors = np.tile(vertex_colors[:, np.newaxis], (1, 3))
  mesh.vertex_colors = o3du.Vector3dVector(vertex_colors)
  return mesh


def calc_nnbr_distance_stats(geom, tree=None, self_tree=True):
  if isinstance(geom, o3dg.PointCloud):
    points = np.asarray(geom.points)
  elif isinstance(geom, o3dg.TriangleMesh):
    points = np.asarray(geom.vertices)
  if tree is None:
    tree = o3dg.KDTreeFlann(geom)
    self_tree = True
  n_nbrs = 2 if self_tree else 1
  distances = []
  for p in points:
    _, _, dist_vector = tree.search_knn_vector_3d(p, n_nbrs)
    distances.append(np.sqrt(dist_vector[n_nbrs-1]))
  return np.mean(distances), np.max(distances)


####### triangle geometry functions
def vertex2trianges(mesh):
  from collections import defaultdict
  v2t = defaultdict(list)
  for i, f in enumerate(np.array(mesh.triangles)):
    for ff in f:
      v2t[ff].append(i)
  return v2t


def mesh_triangle_vertices(mesh):
  vertices = np.array(mesh.vertices)
  triangles = np.array(mesh.triangles)
  v = np.asarray([vertices[f] for f in triangles])
  return v


def triangle_circumcenter_old(ts):
  """
  ts = N x 3 x 3
  from https://gamedev.stackexchange.com/a/60631
  this function has a bug, probably in the order of cross products
  """
  a, b, c = ts[:, 0], ts[:, 1], ts[:, 2]
  ac = c - a
  ab = b - a
  abXac = np.cross(ab, ac, axis=1)
  a_to_cc = (np.cross(abXac, ab, axis=1) * np.sum(ab**2, axis=1, keepdims=True) +\
    np.cross(abXac, ac, axis=1) * np.sum(ac**2, axis=1, keepdims=True)) /\
      (2.0 * np.sum(abXac**2, axis=1, keepdims=True))
  cc_r = np.linalg.norm(a_to_cc, axis=1, keepdims=True)
  cc_c = a + a_to_cc
  return cc_c, cc_r


def triangle_circumcenter(ts):
  a, b, c = ts[:, 0], ts[:, 1], ts[:, 2]
  alpha = a - c
  beta  = b - c
  axb = np.cross(b-a, c-a, axis=1)
  cc_r = (np.linalg.norm(alpha-beta, axis=1, keepdims=True) /
          (2 * np.linalg.norm(axb, axis=1, keepdims=True)) *
          np.linalg.norm(alpha, axis=1, keepdims=True) *
          np.linalg.norm(beta, axis=1, keepdims=True))
  cc_c = np.cross(
    np.sum(alpha**2, axis=1, keepdims=True)*beta -
    np.sum(beta**2,  axis=1, keepdims=True)*alpha,
    np.cross(alpha, beta, axis=1), axis=1
  )
  cc_c /= (2 * np.sum(axb**2, axis=1, keepdims=True))
  return c+cc_c, cc_r


def in_cone(v, os, ns, rs, sigma=0.5):
  """
  v = 3
  os = N x 3
  ns = N x 3
  rs = N
  vectorized implementation of Eq. 10 in https://arxiv.org/pdf/1506.02178.pdf 
  """
  v = v[np.newaxis, :]
  ov = os - v
  d_normal = np.sum(ns * ov, axis=1, keepdims=True)
  d_tangential = np.linalg.norm(ov - ns*d_normal, axis=1, keepdims=True)
  phi = d_tangential / (rs - d_normal*rs/sigma + 1e-6)

  quad_penalty = (-(1.0 - 2.0 * sigma) / (4.0 * sigma ** 2) *
                    (d_normal ** 2) - 1 / (2.0 * sigma) * d_normal +
                    0.25 * (3 - 2 * sigma))
  linear_region_mask = np.logical_and(d_normal <= -sigma, d_normal > -500)
  quad_region_mask   = np.abs(d_normal) < sigma
  repulsion = linear_region_mask*(-d_normal + 1 - sigma) + quad_region_mask*quad_penalty
  out = (phi <= 1) * repulsion

  return out


def test_circumcenter():
  src = o3dg.TriangleMesh.create_box()
  src.compute_triangle_normals()
  src.compute_vertex_normals()

  o, r = triangle_circumcenter(mesh_triangle_vertices(src))
  n = np.array(src.triangle_normals)
  idx = np.random.choice(len(o))
  h = 0.2
  c = o3dg.TriangleMesh.create_cylinder(r[idx], h)
  c.compute_vertex_normals()
  c.translate([0, 0, -h/2])
  T = rotmat_from_vecs(n[idx])
  T[:3, 3] = o[idx]
  c.transform(T)
  o3dv.draw_geometries([src, c]) 


def contact_meshes(src, dst, search_r=10e-3):
  """
  computes contact in [0, 1] for each vertex in dst
  from https://arxiv.org/pdf/1506.02178.pdf Eq. 10
  """
  o, r = triangle_circumcenter(mesh_triangle_vertices(src))
  src_tree = o3dg.KDTreeFlann(src)
  src_v2t = vertex2trianges(src)
  dst_vs = np.array(dst.vertices)
  src.compute_triangle_normals()
  n = -np.array(src.triangle_normals)

  # idx = np.random.choice(len(o))
  # h = 10e-3
  # c = o3dg.TriangleMesh.create_cylinder(r[idx], h)
  # c.compute_vertex_normals()
  # c.translate([0, 0, h/2])
  # T = rotmat_from_vecs(n[idx])
  # T[:3, 3] = o[idx]
  # c.transform(T)
  # o3dv.draw_geometries([src, c]) 
  
  src_ts = []
  for dv in dst_vs:
    k, src_idx, dist2 = src_tree.search_radius_vector_3d(dv, search_r)
    st = []
    for si in src_idx:
      st.extend(src_v2t[si])
    src_ts.append(list(set(st)))

  phi = []
  for dv, st in zip(dst_vs, src_ts):
    if len(st) == 0:
      phi.append(False)
      continue

    ps = in_cone(dv, o[st], n[st], r[st])
    phi.append(np.max(ps))

  return np.asarray(phi)
 

if __name__ == '__main__':
  import os
  osp = os.path
  logging.basicConfig(level=logging.INFO)
  session = 'full28_use'
  object_name = 'camera'
  mesh_filename = 'data/contactpose_data/{:s}/{:s}/{:s}.ply'.format(
      session, object_name, object_name)
  mesh = o3dio.read_triangle_mesh(mesh_filename)
  mesh.compute_vertex_normals()
  mesh_avg_dist, mesh_max_dist = calc_nnbr_distance_stats(mesh)
  mesh_avg_dist *= 1000.0
  mesh_max_dist *= 1000.0
  print('Mesh avg = {:f}, max = {:f} mm'.format(mesh_avg_dist, mesh_max_dist))
  # draw_geometries_with_colormap([mesh])
  
  # # point cloud contact map
  # pc = mesh2pc(mesh, 5000)
  # pc_colors = transfer_color_mesh2pc(mesh, pc)
  # pc = create_pc(pc, pc_colors)
  # draw_geometries_with_colormap([pc])
  # avg_dist, max_dist = calc_nnbr_distance_stats(pc)
  # avg_dist *= 1000.0
  # max_dist *= 1000.0
  # print('PC avg = {:f}, max = {:f} mm'.format(avg_dist, max_dist))
  # pc_mesh = transfer_color_pc2mesh(pc, mesh)
  # draw_geometries_with_colormap([pc_mesh])
  
  # voxel grid contact map
  binvox_filename = osp.join(
      'data', 'binvoxes', '{:s}_hollow.binvox'.format(object_name))
  bpc = binvox2pc(binvox_filename)
  bpc_colors = transfer_color_mesh2pc(mesh, bpc)
  bpc_normals = transfer_normals_mesh2pc(mesh, bpc)
  bpc = create_pc(bpc, bpc_colors)
  bpc.normals = o3du.Vector3dVector(bpc_normals)
  draw_geometries_with_colormap([bpc])
  avg_dist, max_dist = calc_nnbr_distance_stats(bpc)
  avg_dist *= 1000.0
  max_dist *= 1000.0
  print('BPC avg = {:f}, max = {:f} mm'.format(avg_dist, max_dist))
  bpc_mesh = transfer_color_pc2mesh(bpc, mesh)
  draw_geometries_with_colormap([bpc_mesh])
