import logging
import sys
from datetime import datetime
import os
import json
import numpy as np
osp = os.path

def setup_logging(filename=None, noinfo=False):
  handlers = []
  handlers.append(logging.StreamHandler(sys.stdout))
  if filename is not None:
    time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    front, back = osp.split(filename)
    name, ext = back.split('.')
    back = '{:s}_{:s}.{:s}'.format(name, time, ext)
    filename = osp.join(front, back)
    handlers.append(logging.FileHandler(filename, 'w'))

  level = logging.WARN if noinfo else logging.INFO  
  logging.basicConfig(level=level, handlers=handlers)

  if filename is not None:
    root = logging.getLogger()
    root.info('Logging to {:s}'.format(filename))


def load_joint_annotations(p_id, object_name,
    contactpose_dir='data/contactpose_data'):
  filename = osp.join(contactpose_dir, p_id, object_name,
      'annotations.json')
  with open(filename, 'r') as f:
    annotations = json.load(f)

  hands = {}
  for hand_idx, hand in enumerate(annotations['hands']):
    if not hand['valid']:
      continue
    hands[hand_idx] = np.asarray(hand['joints'])

  return hands


def show_annotations(p_id, object_name, contactpose_dir='data/contactpose_data'):
  from open3d import visualization as o3dv
  from open3d import io as o3dio
  from utils import geometry
  
  filename = osp.join(contactpose_dir, p_id, object_name,
      '{:s}.ply'.format(object_name))
  mesh = o3dio.read_triangle_mesh(filename)
  mesh.compute_vertex_normals()
  mesh = geometry.apply_colormap(mesh)
  geoms = [mesh]

  hand_colors = [[0, 1, 0], [1, 0, 0]]
  hands = load_joint_annotations(p_id, object_name, contactpose_dir)
  for hand_idx, hand in hands.items():
    geoms.extend(geometry.create_hand_geoms(
        hand, joint_color=hand_colors[hand_idx]))
  o3dv.draw_geometries(geoms)
