import numpy as np
from open3d import io as o3dio
import os
import logging
from utils.geometry import texture_proc
osp = os.path

def calc_contact_bins(data_dir, hists_filename, l=0.6):
  bins = np.linspace(0, 1, 11)
  
  if osp.isfile(hists_filename):
    hists = np.load(hists_filename)
  else:
    # calculate contact histograms used for contact bin weights
    hists = []
    for session_name in next(os.walk(data_dir))[1]:
      if 'full' not in session_name:
        continue
      print('Session {:s}'.format(session_name))
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
    np.save(hists_filename, hists)
    print('Contact histograms saved to {:s}'.format(hists_filename))
  
  pbar = np.sum(hists, axis=0).astype(float)
  pbar /= sum(pbar)
  w = l * pbar + (1-l) * np.ones(len(pbar)) / len(pbar)
  w = 1 / w
  w /= np.dot(w, pbar)

  filename = osp.join('data', 'texture_bin_weights.npy')
  np.save(filename, w)
  print('Contact bin weights saved to {:s}'.format(filename))


if __name__ == '__main__':
  calc_texture_bins(osp.join('data', 'contactpose_data'),
      osp.join('data', 'texture_hists.npy'))
