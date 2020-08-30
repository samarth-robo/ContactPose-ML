import numpy as np
from open3d import io as o3dio
import os
import logging
from utils.geometry import texture_proc
osp = os.path

def calc_texture_bins(data_dir, l=0.6, hists_file=None):
  logger = logging.getLogger(__name__)
  bins = np.linspace(0, 1, 11)
  
  if hists_file is not None:
    hists = np.load(hists_file)
  else:
    hists = []
    for session_name in next(os.walk(data_dir))[1]:
      if 'full' not in session_name:
        continue
      logger.info('Session {:s}'.format(session_name))
      for object_name in next(os.walk(osp.join(data_dir, session_name)))[1]:
        if ('palm_print' in object_name) or ('hands' in object_name):
          continue
        mesh_filename = osp.join(data_dir, session_name,
                                object_name, '{:s}.ply'.format(object_name))
        mesh = o3dio.read_triangle_mesh(mesh_filename)
        colors = np.asarray(mesh.vertex_colors)[:, 0]
        colors = texture_proc(colors)
        idx = colors > 0
        hist, _ = np.histogram(colors[idx], bins)
        hists.append(hist)
    hists = np.asarray(hists)
  pbar = np.sum(hists, axis=0).astype(float)
  pbar /= sum(pbar)
  w = l * pbar + (1-l) * np.ones(len(pbar)) / len(pbar)
  w = 1 / w
  w /= np.dot(w, pbar)

  return w


if __name__ == '__main__':
  from utils.misc import setup_logging
  setup_logging()
  calc_texture_bins('data/contactdb_data', hists_file='texture_hists.npy')