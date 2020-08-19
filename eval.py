from utils import geometry
from utils.misc import setup_logging, load_joint_annotations
from models.pointnet import PointNetPP
from models.mlp import MLPModel
from models.voxnet import VoxNet
from models.losses import TextureGridLoss
from datasets.contact_pose_3d import ContactPose3D, eval_collate_fn
import train_test_splits

import torch
from torch import nn as tnn
from torch.utils.data import DataLoader
from torch_scatter import scatter_mean
import os
import numpy as np
import argparse
import configparser
import json
import pickle

osp = os.path


def annealed_mean(pred, bins, T=0.1):
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
  joint_locs = load_joint_annotations(session_name, object_name)
  geometry.show_prediction(pred, mesh_filename, binvox_filename, joint_locs,
      kept_lines)


def eval(data_dir, contactpose_dir, split, checkpoint_filename, config_filename,
    device_id, show_object=False, joint_droprate=None, save_preds=True,
    output_filename_suffix=None):
  # config
  config = configparser.ConfigParser()
  config.read(config_filename)
  droprate = config['hyperparams'].getfloat('droprate')
  num_workers = config['misc'].getint('num_workers')
  section = config['hyperparams']
  grid_size = section.getint('grid_size')
  n_rotations = section.getint('n_rotations')
  if joint_droprate is None:
    joint_droprate = section.getfloat('joint_droprate')
  else:
    joint_droprate = float(joint_droprate)

  # cuda
  device = 'cuda:{:s}'.format(device_id)

  # load checkpoint
  checkpoint = torch.load(checkpoint_filename)

  # create model
  texture_weights = np.load('data/texture_bin_weights.npy')
  texture_weights = torch.from_numpy(texture_weights).to(dtype=torch.float)
  model_name = osp.split(config_filename)[1].split('.')[0]
  splits = getattr(train_test_splits, 'split_{:s}'.format(split))
  n_surface_features = 20*2
  if 'pointnet' in model_name:
    model = PointNetPP(n_surface_features, len(texture_weights), droprate)
    loss = tnn.CrossEntropyLoss(weight=texture_weights, ignore_index=-1)
  elif 'mlp' in model_name:
    n_hidden_nodes = config['hyperparams'].getint('n_hidden_nodes')
    model = MLPModel(n_surface_features, len(texture_weights), n_hidden_nodes,
        droprate)
    loss_fn = tnn.CrossEntropyLoss(weight=texture_weights, ignore_index=-1)
  elif 'voxnet' in model_name:
    model = VoxNet(n_surface_features+1, len(texture_weights), droprate)
    loss_fn = TextureGridLoss(weight=texture_weights, eval_mode=True)
  else:
    raise NotImplementedError
  dset = ContactPose3D(data_dir, contactpose_dir, grid_size, n_rotations,
      joint_droprate, eval_mode=True, **splits['test'])
  model.load_state_dict(checkpoint.state_dict())
  model.to(device=device)

  # eval loop!
  dloader = DataLoader(dset, batch_size=n_rotations, pin_memory=True,
                       num_workers=num_workers, collate_fn=eval_collate_fn,
                       shuffle=False)
  all_data = []
  for batch_idx, batch in enumerate(dloader):
    # batch = occs, sdatas, ijks, batch, colors, kept_lines
    session_name, object_name, filename = dset.filenames[batch_idx]
    print('{:d} / {:d}: {:s} {:s}'.format(batch_idx+1, len(dloader), session_name,
                                          object_name))
    if 'voxnet' in model_name:
      use_idx = range(5)
      for idx in use_idx:
        batch[idx] = batch[idx].to(device=device, non_blocking=True)
      with torch.no_grad():
        pred = model(*batch[:4])
        # loss = loss_fn(pred, *batch[2:])
      preds = []
      ijks = batch[2].to(dtype=torch.long)
      for b in range(len(pred)):
        this_ijks = ijks[batch[3] == b]
        preds.append(pred[b, :, this_ijks[:, 2],
                          this_ijks[:, 1], this_ijks[:, 0]])
      preds = [p.t().cpu().numpy() for p in preds]
    elif 'pointnet' in model_name:
      use_idx = range(1, 5)
      for idx in use_idx:
        batch[idx] = batch[idx].to(device=device, non_blocking=True)
      batch[2] = batch[2] / grid_size - 0.5
      with torch.no_grad():
        pred = model(*batch[1:4])
        # loss = loss_fn(pred, batch[4])
      preds = [p.cpu().numpy() for p in torch.chunk(pred, n_rotations)]
    elif 'mlp' in model_name:
      use_idx = [1, 4]
      for idx in use_idx:
        batch[idx] = batch[idx].to(device=device, non_blocking=True)
      with torch.no_grad():
        pred = model(batch[1])
        # loss = loss_fn(pred, batch[4])
      preds = [p.cpu().numpy() for p in torch.chunk(pred, n_rotations)]
    # print('{:s} error = {:.4f}'.format(object_name, loss.item()))
    targ = batch[4].cpu().numpy()
    targ = targ[:len(targ)//n_rotations]
    all_data.append([session_name, object_name, preds, targ])

    if show_object is True:
      avg_pred = np.stack(preds).mean(0)
      kept_lines = batch[-1][0].cpu().numpy()
      show_prediction(avg_pred, kept_lines, dset.texture_bins, object_name,
          session_name)

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
  parser.add_argument('--data_dir', default=osp.join('data', 'skeleton_prediction_data'))
  parser.add_argument('--contactpose_dir', default=osp.join('data', 'contactpose_data'))
  parser.add_argument('--split', type=str, required=True,
                      choices=('objects', 'participants', 'overfit'))
  parser.add_argument('--checkpoint_filename', required=True)
  parser.add_argument('--config_filename', required=True)
  parser.add_argument('--device_id', default='0')
  parser.add_argument('--show_object', action='store_true')
  parser.add_argument('--joint_droprate', default=None)
  parser.add_argument('--save', action='store_true')
  parser.add_argument('--output_filename_suffix', default=None)
  args = parser.parse_args()

  eval(osp.expanduser(args.data_dir), osp.expanduser(args.contactpose_dir),
       args.split,
       osp.expanduser(args.checkpoint_filename),
       osp.expanduser(args.config_filename), args.device_id,
       show_object=args.show_object, joint_droprate=args.joint_droprate,
       save_preds=args.save,
       output_filename_suffix=args.output_filename_suffix)
