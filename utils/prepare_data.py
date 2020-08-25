import init_paths
import numpy as np
import os
import logging
import argparse
from thirdparty import binvox_rw
from open3d import io as o3dio
from open3d import utility as o3du
from open3d import visualization as o3dv
import json
from utils import geometry, mano_utils

osp = os.path

def prepare_occupancy_grids(object_name, config, binvoxes_dir, output_dir):
  logger = logging.getLogger(__name__)
  grid_size = config.getint('grid_size')

  # read binvoxes
  filename = osp.join(
      binvoxes_dir, '{:s}_hollow.binvox'.format(object_name))
  with open(filename, 'rb') as f:
    voxels = binvox_rw.read_as_coord_array(f)
  
  # offset to center the object
  ijk = np.copy(voxels.data)
  mid = (np.max(ijk, 1, keepdims=True) + np.min(ijk, 1, keepdims=True)) / 2.0
  offset = (grid_size/2.0 - mid).astype(np.int)
  ijk += offset
  
  # save
  filename = osp.join(
      output_dir, '{:s}_voxel_surface.npy'.format(object_name))
  np.save(filename, ijk)
  logger.info('{:s} written'.format(filename))
  
  # read binvoxes
  filename = osp.join(
      binvoxes_dir, '{:s}_solid.binvox'.format(object_name))
  with open(filename, 'rb') as f:
    voxels = binvox_rw.read_as_coord_array(f)
  ijk = np.copy(voxels.data) + offset

  # save
  filename = osp.join(
      output_dir, '{:s}_voxel_occupancy.npy'.format(object_name))
  np.save(filename, ijk)
  logger.info('{:s} written'.format(filename))


def prepare_surface_normals(object_name, binvoxes_dir, models_dir, output_dir):
  logger = logging.getLogger(__name__)

  # read binvoxes
  filename = osp.join(
      binvoxes_dir, '{:s}_hollow.binvox'.format(object_name))
  with open(filename, 'rb') as f:
    voxels = binvox_rw.read_as_coord_array(f)
  xyz = geometry.binvox2pc(voxels)

  # read mesh
  mesh_filename = osp.join(models_dir, '{:s}.ply'.format(object_name))
  mesh = o3dio.read_triangle_mesh(mesh_filename)
  if mesh.is_empty():
    logger.warning('{:s} is empty'.format(mesh_filename))
    return
  mesh.scale(1e-3, center=False)
  mesh.compute_vertex_normals()

  # transfer normals
  normals = geometry.transfer_normals_mesh2pc(mesh, xyz)
  # fix surface normals for wine_glass because of thin double sided surface
  if object_name == 'wine_glass':
    pc_center = np.mean(xyz, axis=0, keepdims=True)
    v = pc_center - xyz
    dirs = np.sum(v * normals, axis=1)
    normals[dirs > 0] *= -1
  
  # save
  filename = osp.join(output_dir, '{:s}_voxel_normals.npy'.format(object_name))
  np.save(filename, normals.T)
  logger.info('{:s} written'.format(filename))


def prepare_surface_data(object_name, config, data_dir, binvoxes_dir, output_dir,
                         debug_mode=False):
  logger = logging.getLogger(__name__)
  sigmoid_a = config.getfloat('color_sigmoid_a')
  max_joint_dist = config.getfloat('max_joint_dist_cm') * 1e-2
  n_mano_vertices = 778
  n_hand_joints = 21

  # read binvoxes
  voxels = {}
  for suffix in ['solid', 'hollow']:
    filename = osp.join(
        binvoxes_dir, '{:s}_{:s}.binvox'.format(object_name, suffix))
    with open(filename, 'rb') as f:
      voxels[suffix] = binvox_rw.read_as_coord_array(f)
  
  # occupancy data for checking
  ijk = voxels['solid'].data
  occupancy_set = {tuple(ijk[:, i]) for i in range(ijk.shape[1])}

  # surface data
  # read voxels
  ijk = voxels['hollow'].data
  for p in ijk.T:
    assert(tuple(p) in occupancy_set)
  
  # read mesh
  mesh_filename = osp.join(data_dir, '{:s}.ply'.format(object_name))
  mesh = o3dio.read_triangle_mesh(mesh_filename)
  if mesh.is_empty():
    logger.warning('{:s} is empty'.format(mesh_filename))
    return float('inf')
  mesh.compute_vertex_normals()

  # transfer colors and normals from mesh
  xyz = geometry.binvox2pc(voxels['hollow'])
  colors = geometry.transfer_color_mesh2pc(mesh, xyz)
  colors = geometry.texture_proc(colors, a=sigmoid_a)
  if debug_mode:
    geometry.draw_geometries_with_colormap([geometry.create_pc(xyz, colors)])

  # read oTh in case the hand is moving w.r.t. object
  filename = osp.join(data_dir, 'annotations.json')
  with open(filename, 'r') as f:
    ann = json.load(f)
  oTh = [np.eye(4), np.eye(4)]
  frame_idx = np.random.choice(len(ann['frames']))
  for hand_idx, hand in enumerate(ann['hands']):
    if hand['valid']:
      if hand['moving']:
        frame = ann['frames'][frame_idx]
        oTh[hand_idx] = np.linalg.inv(geometry.pose_matrix(frame['hTo'][hand_idx]))

  # read MANO hands
  filename = osp.join(data_dir, 'mano_fits_15.json')
  try:
    hands = mano_utils.ManoLoader().load_mano(filename, oTh=tuple(oTh))
  except FileNotFoundError as e:
    logger.warning(e)
    return float('inf')

  dists = []
  joint_dists = []
  vectors = []
  for hand in hands:
    if hand is None:
      dists.append(max_joint_dist * np.ones((len(xyz), 1)) + 1)
      vectors.append(np.zeros((len(xyz), 3)))
      joint_dists.append(max_joint_dist * np.ones((len(xyz), n_hand_joints)) + 1)
    else:
      v = hand['vertices'][np.newaxis, :, :] - xyz[:, np.newaxis, :]
      d = np.linalg.norm(v, axis=2)
      idx = np.argsort(d, axis=1)[:, :20]  # smooth point on the hand
      v = np.asarray([np.mean(v[i, idx[i]], axis=0) for i in range(len(v))])
      vectors.append(v)
      d = np.asarray([np.mean(d[i, idx[i]]) for i in range(len(d))])
      d = np.clip(d[:, np.newaxis], 0, max_joint_dist)
      dists.append(d)
      vj = hand['joints'][np.newaxis, :, :] - xyz[:, np.newaxis, :]
      joint_dists.append(np.linalg.norm(vj, axis=2))
  hand_idx = dists[0] > dists[1]
  dists = (1 - hand_idx) * dists[0] + hand_idx * dists[1]
  vectors = (1 - hand_idx) * vectors[0] + hand_idx * vectors[1]
  joint_dists = (1 - hand_idx) * joint_dists[0] + hand_idx * joint_dists[1]

  data = np.hstack((colors[:, np.newaxis], hand_idx, vectors, dists, joint_dists))
  filename = osp.join(
      output_dir, '{:s}_voxel_surface_data.npy'.format(object_name))
  np.save(filename, data)
  logger.info('{:s} written'.format(filename))


def prepare_all_data(session_nums, config, binvoxes_dir,
                     contactpose_data_dir, output_dir, models_dir,
                     debug_mode):
  logger = logging.getLogger(__name__)
  exclude_objects = ['palm_print', 'hands']
  all_object_names = set()
  for session_num in session_nums:
    for instruction in ['use', 'handoff']:
      session_name = 'full{:s}_{:s}'.format(session_num, instruction)
      data_dir = osp.join(contactpose_data_dir, session_name)
      this_output_dir = osp.join(output_dir, session_name)
      if not osp.isdir(this_output_dir):
        os.makedirs(this_output_dir)
      for object_name in next(os.walk(data_dir))[1]:
        skip = object_name in exclude_objects
        for suffix in ['solid', 'hollow']:
          filename = '{:s}_{:s}.binvox'.format(object_name, suffix)
          filename = osp.join(binvoxes_dir, filename)
          if not osp.isfile(filename):
            logger.warning('{:s} binvox does not exist'.format(object_name))
            skip = True
            break
        if skip:
          continue
        all_object_names.add(object_name)
        this_data_dir = osp.join(data_dir, object_name)
        prepare_surface_data(object_name, config,
                             this_data_dir, binvoxes_dir,
                             this_output_dir, debug_mode)

  this_output_dir = osp.join(output_dir, 'normals')
  if not osp.isdir(this_output_dir):
    os.makedirs(this_output_dir)
  for object_name in all_object_names:
    prepare_surface_normals(object_name, binvoxes_dir,
                            models_dir, this_output_dir)
  this_output_dir = osp.join(output_dir, 'occupancy_grids')
  if not osp.isdir(this_output_dir):
    os.makedirs(this_output_dir)
  for object_name in all_object_names:
    prepare_occupancy_grids(object_name, config, binvoxes_dir, this_output_dir)


if __name__ == '__main__':
  import sys
  from utils.misc import setup_logging
  import configparser
  log_filename = osp.split(sys.argv[0])[1].split('.')[0]
  log_filename = osp.join('logs', '{:s}.txt'.format(log_filename))
  setup_logging(log_filename)
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--session_nums', required=True)
  parser.add_argument('--config_filename', default=osp.join('configs', 'mlp.ini'))
  parser.add_argument(
      '--contactpose_data_dir', default=osp.join('data', 'contactpose_data'))
  parser.add_argument(
      '--binvoxes_dir', default=osp.join('data', 'binvoxes'))
  parser.add_argument(
      '--output_dir', default=osp.join('data', 'mesh_prediction_data'))
  parser.add_argument(
      '--models_dir', default=osp.join('data', 'object_models'))
  parser.add_argument('--debug', action='store_true')
  args = parser.parse_args()

  config = configparser.ConfigParser()
  config.read(args.config_filename)
  config = config['hyperparams']

  session_nums = args.session_nums
  if '-' in session_nums:
    start, end = session_nums.split('-')
    session_nums = ['{:d}'.format(i) for i in range(int(start), int(end)+1)]
  else:
    session_nums = session_nums.split(',')
  prepare_all_data(session_nums, config,
                   osp.expanduser(args.binvoxes_dir),
                   osp.expanduser(args.contactpose_data_dir),
                   osp.expanduser(args.output_dir),
                   osp.expanduser(args.models_dir),
                   args.debug)
