if __name__ == '__main__':
  import init_paths
import numpy as np
import transforms3d.quaternions as txq
import logging
from open3d import io as o3dio
from open3d import geometry as o3dg
from open3d import utility as o3du
from open3d import visualization as o3dv
from thirdparty import binvox_rw
from copy import deepcopy
import matplotlib.pyplot as plt


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


def softmax(x):
  return np.exp(x) / sum(np.exp(x))


def create_hand_geoms(joint_locs, color, kept_joints=None):
  geoms = []
  
  if kept_joints is None:
    kept_joints = [True for _ in range(len(joint_locs))]
  pc = o3dg.PointCloud()
  pc.points = o3du.Vector3dVector(joint_locs)
  pc_colors = [color if kept else [0, 0, 0] for kept in kept_joints]
  pc.colors = o3du.Vector3dVector(np.asarray(pc_colors))
  geoms.append(pc)

  ls = o3dg.LineSet()
  ls.points = pc.points
  line_ids = get_hand_line_ids()
  ls.lines = o3du.Vector2iVector(line_ids)
  line_colors = np.zeros((len(line_ids, 3)))
  ls.colors = o3du.Vector3dVector(np.asarray(line_colors))
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
  geoms = [mesh]
  hand_colors = [[0, 1, 0], [1, 0, 0]]
  assert(len(joint_locs) == 2)
  for hand_idx, hand in enumerate(joint_locs):
    if abs(np.max(hand)) < 1e-3:
      continue
    geoms.extend(create_hand_geoms(hand, hand_colors[hand_idx],
      kept_joints[hand_idx]))
  draw_geometries_with_colormap(geoms, apply_sigmoid=False)


def get_hand_line_ids():
  line_ids = []
  for finger in range(5):
      base = 4*finger + 1
      line_ids.append([0, base])
      for j in range(3):
          line_ids.append([base+j, base+j+1])
  line_ids = np.asarray(line_ids, dtype=int)
  return line_ids


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


def draw_geometries_with_colormap(geoms, apply_sigmoid=True):
  logger = logging.getLogger('__name__')

  draw_geoms = []
  for geom in geoms:
    geom = deepcopy(geom)
    if isinstance(geom, o3dg.PointCloud):
      colors = np.asarray(geom.colors)[:, 0]
    elif isinstance(geom, o3dg.TriangleMesh):
      colors = np.asarray(geom.vertex_colors)[:, 0]
    else:
      logger.error('unknown geometry type')
      return
    if apply_sigmoid:
      colors = texture_proc(colors)
    colors = plt.cm.inferno(colors)[:, :3]
    colors = o3du.Vector3dVector(colors)
    if isinstance(geom, o3dg.PointCloud):
      geom.colors = colors
    elif isinstance(geom, o3dg.TriangleMesh):
      geom.vertex_colors = colors
    draw_geoms.append(geom)
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


def create_pc(points, colors=None):
  pc = o3dg.PointCloud()
  pc.points = o3du.Vector3dVector(points)
  if colors is not None:
    colors = np.tile(colors[:, np.newaxis], (1, 3))
    pc.colors = o3du.Vector3dVector(colors)
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
    n_nbrs, idx_nbrs, dist2_nbrs = tree.search_radius_vector_3d(
        p, interp_dist_thresh)
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


if __name__ == '__main__':
  import os
  osp = os.path
  logging.basicConfig(level=logging.INFO)
  session = 'full50_handoff'
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
