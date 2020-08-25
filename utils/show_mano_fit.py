"""
collects the data needed to tune proximity thresholds for contact prediction
"""
import numpy as np
import os
from .import_open3d import *
from . import geometry
from thirdparty.mano.webuser.utils import load_mano
import transforms3d.euler as txe
import matplotlib.pyplot as plt

osp = os.path


def animate(geoms, suffix=None):
  # adjust this transform as needed
  T = np.eye(4)
  T[:3, :3] = txe.euler2mat(np.deg2rad(-90), np.deg2rad(0), np.deg2rad(0))
  for i in range(len(geoms)):
    geoms[i].transform(T)

  animate.count = -1
  animate.step = 50.0  # simulates mouse cursor movement by 50 pixels
  animate.radian_per_pixel = 0.003

  def move_forward(vis):
    glb = animate
    ctr = vis.get_view_control()
    ro = vis.get_render_option()
    ro.point_size = 25.0
    if glb.count >= 0:
      image = vis.capture_screen_float_buffer(False)
      im_filename = osp.join('animation_images',
          'image_{:03d}'.format(glb.count))
      if suffix is not None:
        im_filename = '{:s}_{:s}'.format(im_filename, str(suffix))
      im_filename += '.png'
      plt.imsave(im_filename, np.asarray(image))

      if np.rad2deg(glb.radian_per_pixel * glb.step * glb.count) >= 360.0:
        vis.register_animation_callback(None)

      ctr.rotate(glb.step, 0)
    else:
      # no effect, adjust as needed. Higher values cause the view to zoom out
      ctr.scale(2)  
    glb.count += 1

  o3dv.draw_geometries_with_animation_callback(geoms, move_forward)


def show(p_id, object_name, data_dir,
         hand_color=np.asarray([224.0, 172.0, 105.0])/255,):
  # read object mesh
  filename = osp.join(data_dir, p_id, object_name,
                      '{:s}.ply'.format(object_name))
  object_mesh = o3dio.read_triangle_mesh(filename)
  object_mesh.compute_vertex_normals()
  
  # read mano
  data_dir = osp.join(data_dir, p_id, object_name)
  filename = osp.join(data_dir, 'mano_fits_15.json')
  hands = load_mano(filename)

  hand_meshes = []
  for hand in hands:
    if hand is None:
      continue
    h = o3dg.TriangleMesh()
    h.vertices = o3du.Vector3dVector(hand['vertices'])
    h.triangles = o3du.Vector3iVector(hand['faces'])
    h.compute_vertex_normals()
    h.paint_uniform_color(hand_color)
    hand_meshes.append(h)

  object_mesh = geometry.apply_colormap(object_mesh, apply_sigmoid=True)
  o3dv.draw_geometries(hand_meshes + [object_mesh])
  # object_mesh.paint_uniform_color(plt.cm.tab10(0)[:3])
  # animate(hand_meshes + [object_mesh])


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--p_id', required=True)
  parser.add_argument('--object_name', required=True)
  parser.add_argument('--data_dir', default=osp.join('~', 'contactdb_v2_data'))
  args = parser.parse_args()
  
  show(args.p_id, args.object_name, osp.expanduser(args.data_dir))
