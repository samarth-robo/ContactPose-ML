import os
from import_open3d import *
import numpy as np
import transforms3d.quaternions as txq
import json
import pickle
from utils.geometry import pose_matrix, tform_points

osp = os.path

# Import MANO code
from thirdparty.mano.webuser.smpl_handpca_wrapper_HAND_only \
    import load_model as load_mano_model
# hacks needed for MANO Python2 code
import _pickle as cPickle
import sys
sys.modules['cPickle'] = cPickle
sys.path.append(osp.join('thirdparty', 'mano'))
sys.path.append(osp.join('thirdparty', 'mano', 'webuser'))


def openpose2mano(o, n_joints_per_finger=4):
    finger_o2m = {0: 4, 1: 0, 2: 1, 3: 3, 4: 2}
    m = np.zeros((5*n_joints_per_finger+1, 3))
    m[0] = o[0]
    for ofidx in range(5):
        for jidx in range(n_joints_per_finger):
            oidx = 1 + ofidx*4 + jidx
            midx = 1 + finger_o2m[ofidx]*n_joints_per_finger + jidx
            m[midx] = o[oidx]
    return np.array(m)


# m2o
# 0->1, 1->2, 2->4, 3->3, 4->0
def mano2openpose(m, n_joints_per_finger=4):
    finger_o2m = {0: 4, 1: 0, 2: 1, 3: 3, 4: 2}
    finger_m2o = {v: k for k,v in finger_o2m.items()}
    o = np.zeros((5*n_joints_per_finger+1, 3))
    o[0] = m[0]
    for mfidx in range(5):
        for jidx in range(n_joints_per_finger):
            midx = 1 + mfidx*4 + jidx
            oidx = 1 + finger_m2o[mfidx]*n_joints_per_finger + jidx
            o[oidx] = m[midx]
    return o


def get_palm_joints(p, n_joints_per_finger=4):
    idx = [0]
    for fidx in range(5):
        idx.append(1 + fidx*n_joints_per_finger)
    return p[idx]


def mano_joints_with_fingertips(m):
  fingertip_idxs = [333, 444, 672, 555, 745]
  out = [m.J_transformed[0]]
  for fidx in range(5):
    for jidx in range(4):
      if jidx < 3:
        idx = 1 + fidx*3 + jidx
        out.append(m.J_transformed[idx])
      else:
        out.append(m[fingertip_idxs[fidx]])
  return out


class ManoLoader(object):
  _model_dicts = []
  models_dir = models_dir=osp.join('thirdparty', 'mano', 'models')
  for hand_name in ('LEFT', 'RIGHT'):
    filename = osp.join(models_dir, 'MANO_{:s}.pkl'.format(hand_name))
    with open(filename, 'rb') as f:
      _model_dicts.append(pickle.load(f, encoding='latin1'))
    
    def load_mano(self, params_filename, oTh=(np.eye(4), np.eye(4)),
                  flat_hand_mean=False):
      with open(params_filename, 'r') as f:
        mano_params = json.load(f)
      
      out = []
      for hand_idx, mp in enumerate(mano_params):
        if not mp['valid']:
          out.append(None)
          continue

        ncomps = len(mp['pose']) - 3
        m = load_mano_model(ManoLoader._model_dicts[hand_idx], ncomps=ncomps,
                            flat_hand_mean=flat_hand_mean)
        m.betas[:] = mp['betas']
        m.pose[:]  = mp['pose']

        hTm = np.linalg.inv(pose_matrix(mp['mTc']))
        oTm = oTh[hand_idx] @ hTm

        vertices = np.array(m)
        vertices = tform_points(oTm, vertices)

        joints = mano2openpose(mano_joints_with_fingertips(m))
        joints = tform_points(oTm, joints)

        out.append({
          'vertices': vertices,
          'joints': joints,
          'faces': np.asarray(m.f),
        })
      return out