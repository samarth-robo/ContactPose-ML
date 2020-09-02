from utils import geometry
from utils.misc import setup_logging
from models.image_pointnet import ImagePointNet
from datasets.contact_pose_images import ContactPoseImages, rgb_collate_fn,\
    rgbd_collate_fn
import train_test_splits

import torch
from torch import nn as tnn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import numpy as np
import argparse
import configparser
import json
import pickle
import matplotlib.pyplot as plt

osp = os.path


def annealed_mean(pred, bins, T=0.38):
  pred = np.exp(np.log(pred) / T)
  pred /= pred.sum(axis=1, keepdims=True)
  texture_bin_centers = (bins[:-1] + bins[1:]) / 2.0
  pred = np.sum(pred * texture_bin_centers, axis=1)
  return pred

def mode(pred, bins):
  texture_bin_centers = (bins[:-1] + bins[1:]) / 2.0
  pred = np.argmax(pred, axis=1)
  return texture_bin_centers[pred]


def show_prediction(pred, kept_lines, bins, object_name, session_name,
    models_dir='data/object_models', contactpose_dir='data/contactpose_data',
    binvoxes_dir='data/binvoxes'):
  # create texture
  pred = np.exp(pred) / np.sum(np.exp(pred), axis=1, keepdims=True)
  pred = annealed_mean(pred, bins)
  # pred = mode(pred, bins)
  
  mesh_filename = osp.join(models_dir, '{:s}.ply'.format(object_name))
  binvox_filename = osp.join(binvoxes_dir,
      '{:s}_hollow.binvox'.format(object_name))
  annotation_filename = osp.join(contactpose_dir, session_name, object_name,
      'annotations.json')
  with open(annotation_filename, 'r') as f:
    annotations = json.load(f)
  joint_locs = {}
  for hand_idx, hand in enumerate(annotations['hands']):
    if hand['valid']:
      joint_locs[hand_idx] = np.asarray(hand['joints'])
  geometry.show_prediction(pred, mesh_filename, binvox_filename, joint_locs,
      kept_lines)


def eval(data_dir, contactpose_dir, split, checkpoint_filename, config_filename,
    device_id, show_object=False, joint_droprate=None, save_preds=True,
    output_filename_suffix=None, show_frame_idx=-1):
  model_name = osp.split(config_filename)[1].split('.')[0]
  # config
  config = configparser.ConfigParser()
  config.read(config_filename)
  num_workers = config['misc'].getint('num_workers')
  section = config['hyperparams']
  droprate = section.getfloat('droprate')
  grid_size = section.getint('grid_size')
  n_rotations = section.getint('n_rotations')
  if joint_droprate is None:
    joint_droprate = section.getfloat('joint_droprate')
  else:
    joint_droprate = float(joint_droprate)
  if 'images' in model_name:
    used_cameras = []
    for position in ['left', 'right', 'middle']:
      camera_name = 'kinect2_{:s}'.format(position)
      if section.getboolean(camera_name):
        used_cameras.append(camera_name)
    n_video_frames = int(section['n_video_frames'])
    use_depth = section.getboolean('use_depth')
    n_views = section.getint('n_views')
    im_size = section.getint('im_size')

  # cuda
  device = 'cuda:{:s}'.format(device_id)

  # create model
  texture_weights = np.load('data/texture_bin_weights.npy')
  texture_weights = torch.from_numpy(texture_weights).to(dtype=torch.float)
  splits = getattr(train_test_splits, 'split_{:s}'.format(split))
  n_surface_features = 20*2
  if 'images' in model_name:
    n_channels = 4 if use_depth else 3
    model = ImagePointNet(n_surface_features, len(texture_weights), droprate,
        n_channels, len(used_cameras), n_views, training_mode=False)
    loss_fn = tnn.CrossEntropyLoss(weight=texture_weights, ignore_index=-1)
  else:
    raise NotImplementedError
  # load checkpoint
  checkpoint = torch.load(checkpoint_filename, map_location='cpu')
  model.load_state_dict(checkpoint.state_dict())
  model.to(device=device)
  
  if 'images' in model_name:
    dloader_collate_fn = rgb_collate_fn
    rgb_mean, rgb_std = np.loadtxt('data/rgb_pixel_stats.txt')
    rgb_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(rgb_mean, rgb_std)])
    if use_depth:
      dloader_collate_fn = rgbd_collate_fn
      depth_mean, depth_std = np.loadtxt('data/rgb_pixel_stats.txt')
      depth_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(depth_mean, depth_std)])
    else:
      depth_transform = None

    dset = ContactPoseImages(data_dir, contactpose_dir, grid_size, n_rotations,
        use_depth, used_cameras, n_video_frames, im_size, n_views=3,
        rgb_transform=rgb_transform, depth_transform=depth_transform,
        **splits['test'])
  else:
    raise NotImplementedError

  # eval loop!
  dloader = DataLoader(dset, batch_size=n_rotations, pin_memory=True,
                       num_workers=num_workers, collate_fn=dloader_collate_fn,
                       shuffle=False)
  all_data = []
  n_grasps = len(dset) // n_rotations
  if 'images' in model_name:
    n_grasps = n_grasps // n_video_frames
    frame_idxs = np.round(np.linspace(0, n_video_frames-1, n_rotations,
      endpoint=True)).astype(int)
    if show_object:
      if show_frame_idx >= 0:
        frame_idxs = [show_frame_idx]
      else:
        frame_idxs = [np.random.choice(n_video_frames)]
  else:
    frame_idxs = [0]
  for batch_idx in range(n_grasps):
    preds = []  # keep this here
    for frame_idx in frame_idxs:
      if 'images' in model_name:
        # print('Frame {:d}'.format(frame_idx))
        batch_frame_idx = batch_idx*n_video_frames*n_rotations + frame_idx*n_rotations
        batch = dloader_collate_fn([dset[batch_frame_idx+i] for i in range(n_rotations)])
        batch = list(batch)
      else:
        batch = dloader[batch_idx]
      session_name, object_name, filename = dset.filenames[batch_idx]
      print('{:d} / {:d}: {:s} {:s}'.format(batch_idx+1, n_grasps, session_name,
                                            object_name))
      if 'images' in model_name:
        # rgb_images, [depth_images], verts, ijks, batch, colors = batch
        use_idx = [0, -3, -2, -1]
        if use_depth:
          use_idx.append(1)
        for idx in use_idx:
          batch[idx] = batch[idx].to(device=device, non_blocking=True)
        if use_depth:
          batch[0] = torch.cat((batch[0], batch[1]), dim=2)
        with torch.no_grad():
          pred = model(batch[0], *batch[-4:-1])
          # loss = loss_fn(pred, batch[-1])
        pred = [p.cpu().numpy() for p in torch.chunk(pred, n_rotations)]
        preds.append(np.stack(pred).mean(0))
      # print('{:s} error = {:.4f}'.format(object_name, loss.item()))
    targ_idx = 4
    if ('images' in model_name) and use_depth:
      targ_idx = 5
    targ = batch[targ_idx].cpu().numpy()
    targ = targ[:len(targ)//n_rotations]
    all_data.append([session_name, object_name, preds, targ])

    if show_object is True:
      if 'images' in model_name:
        # kept_lines = [None, None]
        # ims = batch[0][0, ...].cpu().numpy()
        # ims = np.transpose(ims, (0, 2, 3, 1))
        # plt.ion()
        # for camera_name, im in zip(['left', 'right', 'middle'], ims):
        #   im = im * rgb_std + rgb_mean
        #   im = np.clip(im, a_min=0, a_max=1)
        #   plt.figure()
        #   plt.imshow(im)
        #   plt.title(camera_name)
        #   plt.pause(0.1)
        # plt.show()
        idx = np.random.choice(len(preds))
        avg_pred = preds[idx]
      else:
        kept_lines = batch[-1][0].cpu().numpy()
        avg_pred = np.stack(preds).mean(0)
      show_prediction(avg_pred, kept_lines, dset.texture_bins, object_name,
          session_name)
      # if 'images' in model_name:
      #   while plt.get_fignums():
      #     plt.pause(0.001)
      #   plt.close('all')

  if save_preds:
    output_dir = osp.split(checkpoint_filename)[0]
    filename = osp.join(output_dir,
                        'predictions_joint_droprate={:.2f}'.format(
                            joint_droprate))
    if output_filename_suffix is not None:
      filename = '{:s}_{:s}'.format(filename, output_filename_suffix)
    filename += '.pkl'
    with open(filename, 'wb') as f:
      pickle.dump({
        'data': all_data,
        'checkpoint_filename': checkpoint_filename,
        'joint_droprate': joint_droprate}, f)
    print('{:s} written'.format(filename))

if __name__ == '__main__':
  from utils.misc import setup_logging
  setup_logging(noinfo=True)

  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir',
                      default=osp.join('data', 'images_prediction_data'))
  parser.add_argument('--contactpose_dir',
                      default=osp.join('data', 'contactpose_data'))
  parser.add_argument('--split', type=str, required=True,
                      choices=('objects', 'participants', 'overfit', 'images_objects',
                        'images_participants'))
  parser.add_argument('--checkpoint_filename', required=True)
  parser.add_argument('--config_filename', required=True)
  parser.add_argument('--device_id', default='0')
  parser.add_argument('--show_object', action='store_true')
  parser.add_argument('--joint_droprate', default=None,
                      help="irrelevant, present for legacy reasons")
  parser.add_argument('--save', action='store_true')
  parser.add_argument('--output_filename_suffix', default=None)
  parser.add_argument('--frame_idx', default=-1, type=int)
  args = parser.parse_args()

  eval(osp.expanduser(args.data_dir), osp.expanduser(args.contactpose_dir),
       args.split,
       osp.expanduser(args.checkpoint_filename),
       osp.expanduser(args.config_filename), args.device_id,
       show_object=args.show_object, joint_droprate=args.joint_droprate,
       save_preds=args.save,
       output_filename_suffix=args.output_filename_suffix,
       show_frame_idx=args.frame_idx)
