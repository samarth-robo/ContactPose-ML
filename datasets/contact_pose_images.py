if __name__ == '__main__':
  from utils import geometry  # import open3d before pytorch because of CXX_ABI
import torch
from torch.utils.data import Dataset
import numpy as np
from utils.misc import setup_logging
import os
import logging
import json
from PIL import Image
import transforms3d.euler as txe
osp = os.path


class ContactPoseImages(Dataset):
  def __init__(self, data_dir, contactpose_data_dir, grid_size, n_rotations,
               use_depth, used_cameras, n_video_frames, im_size, n_views,
               rgb_transform=None, depth_transform=None,
               include_instructions=None,
               include_objects=None,
               include_sessions=None,
               exclude_instructions=None,
               exclude_objects=None,
               exclude_sessions=None):
    super(ContactPoseImages, self).__init__()
    setup_logging()
    self.logger = logging.getLogger(__name__)

    data_dir = osp.expanduser(data_dir)
    contactpose_data_dir = osp.expanduser(contactpose_data_dir)
    self.rgb_transform = rgb_transform
    self.depth_transform = depth_transform
    self.grid_size = grid_size
    self.n_rotations = n_rotations
    self.use_depth = use_depth
    self.used_cameras = used_cameras
    self.n_video_frames = n_video_frames
    self.im_size = im_size
    self.n_views = n_views

    self.texture_bin_weights = np.load('data/texture_bin_weights.npy')
    self.texture_bins = np.linspace(0, 1, len(self.texture_bin_weights)+1)

    # collect various filenames and data
    self.surfaces = {}
    self.filenames = []
    self.rgb_image_filenames = {}
    self.depth_image_filenames = {}
    self.image_verts_filenames = {}
    surfaces_dir = osp.join(data_dir, 'occupancy_grids')
    for filename in next(os.walk(surfaces_dir))[-1]:
      if '_voxel_surface' not in filename:
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

      surface_filename = osp.join(surfaces_dir,
                                  '{:s}_voxel_surface.npy'.format(object_name))
      if not osp.isfile(surface_filename):
        self.logger.warn('{:s} does not exist'.format(surface_filename))
        continue
      self.surfaces[object_name] = np.load(surface_filename)

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
      
      self.rgb_image_filenames[session_name] = {}
      self.depth_image_filenames[session_name] = {}
      self.image_verts_filenames[session_name] = {}
      this_data_dir = osp.join(data_dir, session_name)
      for object_name in self.surfaces.keys():
        surface_filename = osp.join(
            this_data_dir, '{:s}_voxel_surface_data.npy'.format(object_name))
        if not osp.isfile(surface_filename):
          self.logger.info('Skipping {:s} {:s}'.format(
              session_name, object_name))
          continue
        if object_name not in self.rgb_image_filenames[session_name]:
          self.rgb_image_filenames[session_name][object_name] = []
          self.depth_image_filenames[session_name][object_name] = []
          self.image_verts_filenames[session_name][object_name] = []

        images_dir = osp.join(contactpose_data_dir, session_name, object_name,
                              'images')
        if not osp.isdir(images_dir):
          continue
        all_rgb_filenames   = []
        all_depth_filenames = []
        all_vert_filenames  = []
        for frame_idx in range(self.n_video_frames):
          add = True
          rgb_filenames = []
          depth_filenames = []
          vert_filenames = []
          for camera_name in self.used_cameras:
            rfilename = osp.join(images_dir, camera_name, 'color',
                'frame{:03d}.png'.format(frame_idx))
            dfilename = osp.join(images_dir, camera_name, 'depth',
                'frame{:03d}.png'.format(frame_idx))
            pfilename = osp.join(images_dir, camera_name, 'projections',
                'frame{:03d}_verts.npy'.format(frame_idx))
            if osp.isfile(rfilename) and osp.isfile(dfilename) and \
                osp.isfile(pfilename):
              rgb_filenames.append(rfilename)
              depth_filenames.append(dfilename)
              vert_filenames.append(pfilename)
          N = len(rgb_filenames)
          if N == 0:
            continue
          elif N < self.n_views:
            # duplicate
            rgb_filenames   = [rgb_filenames[i%N]   for i in range(self.n_views)]
            depth_filenames = [depth_filenames[i%N] for i in range(self.n_views)]
            vert_filenames  = [vert_filenames[i%N]  for i in range(self.n_views)]
          all_rgb_filenames.append(rgb_filenames)
          all_depth_filenames.append(depth_filenames)
          all_vert_filenames.append(vert_filenames)
        
        if len(all_rgb_filenames) > 0:
          self.filenames.append(
              [session_name, object_name, surface_filename])
          self.rgb_image_filenames[session_name][object_name] = all_rgb_filenames
          self.depth_image_filenames[session_name][object_name] = all_depth_filenames
          self.image_verts_filenames[session_name][object_name] = all_vert_filenames
      
  def __len__(self):
    return len(self.filenames) * self.n_rotations * self.n_video_frames

  def rotate_grid(self, ijk, R):
    p = ijk + 0.5
    p = p - self.grid_size/2.0
    p = R @ p
    p = p + self.grid_size/2.0
    p = np.round(p - 0.5)
    p = np.minimum(np.maximum(p, 0), self.grid_size-1)
    return p

  def __getitem__(self, index):
    filename_idx, rot_frame_idx = divmod(
        index, self.n_rotations*self.n_video_frames)
    frame_idx, rotation_idx = divmod(rot_frame_idx, self.n_rotations)
    session_name, object_name, surface_data_filename = \
        self.filenames[filename_idx]
    frame_idx = frame_idx % \
        len(self.rgb_image_filenames[session_name][object_name])

    theta = np.deg2rad((360.0 / self.n_rotations) * rotation_idx)
    R = txe.euler2mat(0, 0, theta)

    # texture
    sdata = np.load(surface_data_filename)
    colors = sdata[:, 0]
    ijk = self.surfaces[object_name]
    ijk = self.rotate_grid(ijk, R).T

    # bin colors
    idx = np.logical_not(colors > 0)
    colors = np.digitize(colors, self.texture_bins) - 1
    colors[idx] = -1

    # images
    rgb_filenames = \
      self.rgb_image_filenames[session_name][object_name][frame_idx]
    depth_filenames = \
      self.depth_image_filenames[session_name][object_name][frame_idx]
    vert_filenames = \
      self.image_verts_filenames[session_name][object_name][frame_idx]
    rgb_images = [Image.open(f) for f in rgb_filenames]
    if self.rgb_transform is not None:
      rgb_images = [self.rgb_transform(i) for i in rgb_images]
    verts = []
    for f in vert_filenames:
      v = np.load(f).astype(np.int)
      idx = np.logical_and(v[:, 0]>0, v[:, 1]>0)
      idx = np.logical_and(idx, v[:, 0] < self.im_size)
      idx = np.logical_and(idx, v[:, 1] < self.im_size)
      v = v[idx]
      verts.append(v)
    if self.use_depth:
      depth_images = [Image.open(f) for f in depth_filenames]
      if self.depth_transform is not None:
        depth_images = [self.depth_transform(i) for i in depth_images]
      out = rgb_images, depth_images, verts, ijk.astype(np.float32), colors
    else:
      out = rgb_images, verts, ijk.astype(np.float32), colors
    return out


def rgb_collate_fn(data):
  rgb_images = []
  verts = []
  ijks = []
  colors = []
  batch = []
  for idx, d in enumerate(data):
    rgb_images.append(torch.stack(
        [torch.as_tensor(i, dtype=torch.float32) for i in d[0]]))
    verts.append(d[1])
    ijks.append(torch.as_tensor(d[2], dtype=torch.float32))
    colors.append(torch.as_tensor(d[3], dtype=torch.long))
    batch.append(idx * torch.ones(len(d[2]), dtype=torch.long))
  rgb_images = torch.stack(rgb_images)
  ijks = torch.cat(ijks)
  colors = torch.cat(colors)
  batch = torch.cat(batch)
  return rgb_images, verts, ijks, batch, colors


def rgbd_collate_fn(data):
  rgb_images = []
  depth_images = []
  verts = []
  ijks = []
  colors = []
  batch = []
  for idx, d in enumerate(data):
    rgb_images.append(torch.as_tensor(d[0], dtype=torch.float32))
    depth_images.append(torch.as_tensor(d[1], dtype=torch.float32))
    verts.append(d[2])
    ijks.append(torch.as_tensor(d[3], dtype=torch.float32))
    colors.append(torch.as_tensor(d[4], dtype=torch.long))
    batch.append(idx * torch.ones(len(d[3]), dtype=torch.long))
  rgb_images = torch.stack(rgb_images)
  depth_images = torch.stack(depth_images)
  ijks = torch.cat(ijks)
  colors = torch.cat(colors)
  batch = torch.cat(batch)
  return rgb_images, depth_images, verts, ijks, batch, colors


if __name__ == '__main__':
  from torch.utils.data import DataLoader
  from torchvision import transforms
  import matplotlib.pyplot as plt
  use_depth = False
  rgb_transform = transforms.ToTensor()
  dset = ContactPoseImages('data/prediction_data', 'data/contactpose_data', 64, 1,
                       use_depth,
                       rgb_transform=rgb_transform,
                       include_sessions=[36],
                       include_instructions=['handoff'])
  texture_bin_centers = (dset.texture_bins[:-1] + dset.texture_bins[1:]) / 2.0
  N_show = 10
  dloader = DataLoader(dset, batch_size=N_show,
                       collate_fn=rgb_collate_fn, shuffle=True)
  for data in dloader:
    for batch_idx in range(N_show):
      # unpack
      rgb_image = data[0][batch_idx, 0].numpy()
      plt.imshow(np.transpose(rgb_image, (1, 2, 0)))
      plt.show()
      
      pos = data[2][data[-2] == batch_idx].numpy()
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
