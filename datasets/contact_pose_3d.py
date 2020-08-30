if __name__ == '__main__':
  from utils import geometry  # import open3d before pytorch because of CXX_ABI
import torch
from torch.utils.data import Dataset
import numpy as np
from thirdparty import binvox_rw
from utils.misc import setup_logging
import os
import logging
import transforms3d.euler as txe
import json
osp = os.path

class ContactPose3D(Dataset):
  def __init__(self, data_dir, contactpose_data_dir, grid_size, n_rotations,
               joint_droprate, eval_mode=False,
               include_instructions=None,
               include_objects=None,
               include_sessions=None,
               exclude_instructions=None,
               exclude_objects=None,
               exclude_sessions=None):
    super(ContactPose3D, self).__init__()
    setup_logging()
    self.logger = logging.getLogger(__name__)

    data_dir = osp.expanduser(data_dir)
    contactpose_data_dir = osp.expanduser(contactpose_data_dir)
    self.grid_size = grid_size
    self.n_rotations = n_rotations
    self.joint_droprate = joint_droprate
    self.eval_mode = eval_mode
    # ensure same hand segments are dropped for all rotations in eval mode
    self.filenameidx2keptjoints = {}
    assert(self.joint_droprate >= 0 and self.joint_droprate < 1)

    self.texture_bin_weights = np.load('data/texture_bin_weights.npy')
    self.texture_bins = np.linspace(0, 1, len(self.texture_bin_weights)+1)

    # collect normals and occupancy grids
    self.normals = {}
    self.occupancies = {}
    self.surfaces = {}
    normals_dir = osp.join(data_dir, 'normals')
    occupancies_dir = osp.join(data_dir, 'occupancy_grids')
    for filename in next(os.walk(normals_dir))[-1]:
      if '_voxel_normals' not in filename:
        continue
      object_name = filename[:filename.find('_voxel')]
      
      if include_objects is not None:
        if object_name not in include_objects:
          self.logger.info('Skipping object {:s}'.format(object_name))
          continue
      if exclude_objects is not None:
        if object_name in exclude_objects:
          self.logger.info('Skipping object {:s}'.format(object_name))
          continue

      normals_filename = osp.join(normals_dir, filename)
      occupancy_filename = osp.join(
          occupancies_dir, '{:s}_voxel_occupancy.npy'.format(object_name))
      if not osp.isfile(occupancy_filename):
        self.logger.warning('{:s} does not exist'.format(occupancy_filename))
        continue
      surface_filename = osp.join(
          occupancies_dir, '{:s}_voxel_surface.npy'.format(object_name))
      if not osp.isfile(surface_filename):
        self.logger.warning('{:s} does not exist'.format(surface_filename))
        continue
      self.normals[object_name] = np.load(normals_filename)
      self.occupancies[object_name] = np.load(occupancy_filename)
      self.surfaces[object_name] = np.load(surface_filename)
    
    # collect surface data filenames
    self.filenames = []
    self.joint_locations = {}
    for session_name in next(os.walk(data_dir))[1]:
      if 'full' not in session_name:
        continue
      session_num, instruction = session_name.split('_')
      session_num = int(session_num[4:])

      if include_sessions is not None:
        if session_num not in include_sessions:
          self.logger.info('Skipping session {:s}'.format(session_name))
          continue
      if exclude_sessions is not None:
        if session_num in exclude_sessions:
          self.logger.info('Skipping session {:s}'.format(session_name))
          continue
      if include_instructions is not None:
        if instruction not in include_instructions:
          self.logger.info('Skipping session {:s}'.format(instruction))
          continue
      if exclude_instructions is not None:
        if instruction in exclude_instructions:
          self.logger.info('Skipping session {:s}'.format(instruction))
          continue
      
      self.joint_locations[session_name] = {}
      this_data_dir = osp.join(data_dir, session_name)
      for object_name in self.normals.keys():
        filename = osp.join(
            this_data_dir, '{:s}_voxel_surface_data.npy'.format(object_name))
        if not osp.isfile(filename):
          self.logger.info('Skipping {:s} {:s}'.format(
              session_name, object_name))
          continue
        else:
          self.filenames.append([session_name, object_name, filename])

        filename = osp.join(contactpose_data_dir, session_name,
                            object_name, 'annotations.json')
        with open(filename, 'r') as f:
          annotations = json.load(f)
        self.joint_locations[session_name][object_name] = np.asarray(
            [annotations['hands'][i]['joints'] for i in range(2)])
      
  def __len__(self):
    return len(self.filenames) * self.n_rotations

  def rotate_grid(self, ijk, R, grid_size):
    p = ijk + 0.5
    p = p - self.grid_size/2.0
    p = R @ p
    p = p + self.grid_size/2.0
    p = np.round(p - 0.5)
    p = np.minimum(np.maximum(p, 0), grid_size-1)
    return p

  def drop_joints_random(self, N_joints):
    mask = np.random.rand(N_joints, 1)
    mask = mask > self.joint_droprate
    return mask

  def drop_joints_camera(self, oX):
    R = txe.euler2mat(*np.random.uniform(0, 2*np.pi, size=3))
    oX_r = R @ oX.T
    order = np.argsort(oX_r[0])
    drop_joint_idx = order[:int(self.joint_droprate*len(order))]
    mask = np.ones((len(oX), 1)) > 0
    mask[drop_joint_idx, 0] = False
    return mask

  def __getitem__(self, index):
    filename_idx, rotation_idx = divmod(index, self.n_rotations)
    theta = np.deg2rad((360.0 / self.n_rotations) * rotation_idx)
    R = txe.euler2mat(0, 0, theta)
    session_name, object_name, surface_data_filename = \
        self.filenames[filename_idx]

    # occupancy grid
    occ = np.zeros((1, self.grid_size, self.grid_size, self.grid_size),
                   dtype=np.float32)
    ijk = self.occupancies[object_name]
    ijk = self.rotate_grid(ijk, R, self.grid_size).astype(int)
    occ[0, ijk[2], ijk[1], ijk[0]] = 1

    # surface data
    sdata = np.load(surface_data_filename)
    colors, joint_vectors = sdata[:1], sdata[1:]
    colors = np.squeeze(colors)
    ijk = self.surfaces[object_name]
    ijk = self.rotate_grid(ijk, R, self.grid_size).T

    # bin colors
    idx = np.logical_not(colors > 0)
    colors = np.digitize(colors, self.texture_bins) - 1
    colors[idx] = -1
    
    # decide closest hand
    joint_dists = np.vsplit(joint_vectors, 42)
    joint_dists = np.vstack([np.linalg.norm(d, axis=0) for d in joint_dists])
    closest_joint_idx = np.argmin(joint_dists, axis=0)[np.newaxis, :]
    left_hand_closest = closest_joint_idx < 21

    # select features of the closest hand
    joint_vectors = np.vsplit(joint_vectors, 2)
    joint_vectors = left_hand_closest*joint_vectors[0] + \
        (1-left_hand_closest)*joint_vectors[1]

    # drop joints
    # mask = self.drop_joints_random(joint_vectors.shape[0]//3)
    joints = self.joint_locations[session_name][object_name]
    if filename_idx not in self.filenameidx2keptjoints:
      kept_joints = [self.drop_joints_camera(
          joints[i]) for i in range(len(joints))]
    elif self.eval_mode:
      kept_joints = self.filenameidx2keptjoints[filename_idx]
      kept_joints = [(d>0)[:, np.newaxis] for d in kept_joints]
    for i in range(2):
      idx = left_hand_closest.squeeze() == (1 - i)
      mask = np.tile(kept_joints[i], (1, 3)).reshape((-1, 1))
      joint_vectors[:, idx] *= mask

    sdata = np.vstack((self.normals[object_name], joint_vectors)).T

    out = [occ, sdata.astype(np.float32), ijk.astype(np.float32), colors]
    if self.eval_mode:
      kept_joints = [d.squeeze().astype(np.int) for d in kept_joints]
      self.filenameidx2keptjoints[filename_idx] = np.stack(kept_joints)
      out.append(kept_joints)
    return out

def collate_fn(data):
  occs = []
  sdatas = []
  ijks = []
  colors = []
  batch = []
  for idx, d in enumerate(data):
    occs.append(torch.as_tensor(d[0], dtype=torch.float32))
    sdatas.append(torch.as_tensor(d[1], dtype=torch.float32))
    ijks.append(torch.as_tensor(d[2], dtype=torch.float32))
    colors.append(torch.as_tensor(d[3], dtype=torch.long))
    batch.append(idx * torch.ones(len(d[1]), dtype=torch.long))
  occs = torch.stack(occs)
  sdatas = torch.cat(sdatas)
  ijks = torch.cat(ijks)
  colors = torch.cat(colors)
  batch = torch.cat(batch)
  return occs, sdatas, ijks, batch, colors

def eval_collate_fn(data):
  occs = []
  sdatas = []
  ijks = []
  colors = []
  kept_joints = []
  batch = []
  for idx, d in enumerate(data):
    occs.append(torch.as_tensor(d[0], dtype=torch.float32))
    sdatas.append(torch.as_tensor(d[1], dtype=torch.float32))
    ijks.append(torch.as_tensor(d[2], dtype=torch.float32))
    colors.append(torch.as_tensor(d[3], dtype=torch.long))
    kept_joints.append(torch.as_tensor(d[4], dtype=torch.long))
    batch.append(idx * torch.ones(len(d[1]), dtype=torch.long))
  occs = torch.stack(occs)
  sdatas = torch.cat(sdatas)
  ijks = torch.cat(ijks)
  colors = torch.cat(colors)
  kept_joints = torch.stack(kept_joints)
  batch = torch.cat(batch)
  return occs, sdatas, ijks, batch, colors, kept_joints


if __name__ == '__main__':
  from torch.utils.data import DataLoader
  joint_droprate = 0.3
  dset = ContactPose3D('data/prediction_data', 'data/contactpose_data', 64, 12,
                       joint_droprate, include_sessions=[10,])
  texture_bin_centers = (dset.texture_bins[:-1] + dset.texture_bins[1:]) / 2.0
  N_show = 10
  dloader = DataLoader(dset, batch_size=N_show,
                       collate_fn=collate_fn, shuffle=True)
  for data in dloader:
    for batch_idx in range(N_show):
      # unpack
      occ = data[0][batch_idx, 0].numpy()
      pos = data[2][data[-2] == batch_idx].numpy()
      
      # colors = data[1][:, 3:][data[-2] == batch_idx].numpy().squeeze()
      # colors = np.vstack([np.linalg.norm(v, axis=1)
      #                     for v in np.hsplit(colors, 21)])
      # colors = np.min(colors, axis=0)
      # colors /= colors.max()

      colors = data[-1][data[-2] == batch_idx].numpy().squeeze()
      # unquantize colors
      idx = colors < 0
      colors = texture_bin_centers[colors]
      colors[idx] = 0

      # show
      geometry.draw_geometries_with_colormap([geometry.create_pc(pos, colors)])
      # occ_colors = np.zeros(occ.shape)
      # i, j, k = pos.astype(int).T
      # occ_colors[k, j, i] = colors
      # z, y, x = np.nonzero(occ)
      # colors_show = occ_colors[z, y, x]
      # pc = np.vstack((x, y, z)).T
      # geometry.draw_geometries_with_colormap([geometry.create_pc(pc, colors_show)])
    break
