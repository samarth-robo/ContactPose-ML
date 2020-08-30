import init_paths
from open3d import utility as o3du
from open3d import visualization as o3dv
from open3d import geometry as o3dg
from open3d import io as o3dio
import subprocess
import numpy as np
import os
import logging
import argparse
from thirdparty import binvox_rw
import configparser
import transforms3d.euler as txe
osp = os.path


def calc_largest_bb(mesh_filename, N_rotations):
  """
  Calculates the bounding box for voxelization and vTo matrix that transforms
  from object coordinates to voxelgrid coordinates
  """
  mesh = o3dio.read_triangle_mesh(mesh_filename)

  # center the mesh
  o = (mesh.get_min_bound() + mesh.get_max_bound()) / 2.0
  o[2] = 0
  mesh.translate(-o)

  # rotate and check extents
  min_pt = np.asarray(mesh.get_min_bound())
  max_pt = np.asarray(mesh.get_max_bound())
  T = np.eye(4)
  T[:3, :3] = txe.euler2mat(0, 0, np.deg2rad(360.0 / N_rotations))
  for _ in range(N_rotations-1):
    mesh.transform(T)
    min_pt = np.minimum(min_pt, mesh.get_min_bound())
    max_pt = np.maximum(max_pt, mesh.get_max_bound())
  
  # de-center the bounding box
  min_pt += o
  max_pt += o

  return min_pt, max_pt


def run_binvox(mesh_filename, output_filename, N_voxels, min_pt, max_pt, hollow):
  logger = logging.getLogger(__name__)
  
  # voxelize mesh
  args = osp.join('..', 'thirdparty', 'binvox')
  args += ' -pb -d {:d} '.format(N_voxels)
  args += '-bb {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} '.\
    format(*np.hstack((min_pt, max_pt)))
  if hollow:
    args += '-ri '
  args += mesh_filename

  # remove old binvox, because that causes output to another unknown
  # filename
  vox_filename = mesh_filename.replace('.ply', '.binvox')
  if osp.isfile(vox_filename):
    os.remove(vox_filename)
  try:
    subprocess.check_call(args, shell=True)
  except subprocess.CalledProcessError as e:
    logger.error(e)

  os.rename(vox_filename, output_filename)
  logger.info('Written {:s}'.format(output_filename))


def generate_binvoxes(models_dir, output_dir, N_voxels, N_rotations,
                      debug_mode=False):
  logger = logging.getLogger(__name__)

  mesh_filenames = [osp.join(models_dir, fn)
    for fn in next(os.walk(models_dir))[-1] if '.ply' in fn]

  for mesh_filename in mesh_filenames:
    min_pt, max_pt = calc_largest_bb(mesh_filename, N_rotations)

    object_name = mesh_filename.split('/')[-1].split('.')[0]

    for hollow in [True, False]:
      suffix = 'hollow' if hollow else 'solid'
      output_filename = '{:s}_{:s}.binvox'.format(object_name, suffix)
      output_filename = osp.join(output_dir, output_filename)
      run_binvox(mesh_filename, output_filename,
                 N_voxels, min_pt, max_pt, hollow)

    if debug_mode:
      output_filename = '{:s}_solid.binvox'.format(object_name)
      output_filename = osp.join(output_dir, output_filename)
      with open(output_filename, 'rb') as f:
        mv = binvox_rw.read_as_3d_array(f)
      x, y, z = np.where(mv.data)
      pc = o3dg.PointCloud()
      pc.points = o3du.Vector3dVector(np.vstack((x, y, z)).T)
      o3dv.draw_geometries([pc])


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--models_dir', default=osp.join('..', 'data', 'object_models'))
  parser.add_argument(
      '--output_dir', default=osp.join('..', 'data', 'binvoxes'))
  parser.add_argument('--config_file', required=True)
  parser.add_argument('--debug', action='store_true')
  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO)

  config = configparser.ConfigParser()
  config.read(osp.expanduser(args.config_file))
  config = config['hyperparams']
  generate_binvoxes(osp.expanduser(args.models_dir),
                    osp.expanduser(args.output_dir),
                    N_voxels=config.getint('grid_size'),
                    N_rotations=config.getint('n_rotations'),
                    debug_mode=args.debug)
