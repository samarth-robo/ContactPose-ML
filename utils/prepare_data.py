import numpy as np
import os
import logging
import argparse
from thirdparty import binvox_rw
from open3d import io as o3dio
from open3d import utility as o3du
import json

from utils import geometry
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

  # read hand joints
  hand_joints = []
  valid_hands = []
  json_filename = osp.join(data_dir, 'annotations.json')
  try:
    with open(json_filename, 'r') as f:
      annotations = json.load(f)
  except FileNotFoundError as e:
    logger.warning(e)
    return float('inf')
  annotations = annotations['hands']
  for hand in annotations:
    valid_hands.append(hand['valid'])
    hand_joints.append(hand['joints'])
  hand_joints = np.asarray(hand_joints)

  # find closest valid hand
  dists = []
  for this_hand_joints, valid in zip(hand_joints, valid_hands):
    if not valid:
      dists.append(max_joint_dist *
                   np.ones((len(xyz), len(this_hand_joints))) + 1)
    else:
      vectors = this_hand_joints[np.newaxis, :, :] - xyz[:, np.newaxis, :]
      dists.append(np.linalg.norm(vectors, axis=2))
  dists = np.hstack(dists)
  hand_idx = np.argmin(dists, axis=1)
  hand_idx = (hand_idx >= dists.shape[1]//2).astype(np.int)

  # surface features = vectors to closest point on each hand line segment
  hand_line_ids = geometry.get_hand_line_ids()
  vectors = np.zeros((len(xyz), len(hand_line_ids)*3))
  for i in range(2):
    idx = hand_idx == i
    if np.sum(idx) == 0:
      continue
    l0 = hand_joints[i][hand_line_ids[:, 0]]
    l1 = hand_joints[i][hand_line_ids[:, 1]]
    p  = xyz[idx]
    pl = geometry.closest_linesegment_point(l0, l1, p)
    v  = pl - p[:, np.newaxis, :]
    v  = np.reshape(v, (len(p), -1))
    vectors[idx, :] = v

  data = np.hstack((colors[:, np.newaxis], hand_idx[:, np.newaxis], vectors))
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
        os.mkdir(this_output_dir)
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

  # this_output_dir = osp.join(output_dir, 'normals')
  # if not osp.isdir(this_output_dir):
  #   os.mkdir(this_output_dir)
  # for object_name in all_object_names:
  #   prepare_surface_normals(object_name, binvoxes_dir,
  #                           models_dir, this_output_dir)
  # this_output_dir = osp.join(output_dir, 'occupancy_grids')
  # if not osp.isdir(this_output_dir):
  #   os.mkdir(this_output_dir)
  # for object_name in all_object_names:
  #   prepare_occupancy_grids(object_name, config, binvoxes_dir, this_output_dir)


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
      '--output_dir', default=osp.join('data', 'skeleton_prediction_data'))
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
