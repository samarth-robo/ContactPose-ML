from models.pointnet import PointNetPP
from models.mlp import MLPModel
from datasets.contact_pose_3d import ContactPose3D, collate_fn
import train_test_splits
from utils.misc import setup_logging

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
import visdom
from torch.utils.data import DataLoader
from torch import optim as toptim
from torch import nn as tnn
import numpy as np
import os
import torch
import configparser
import argparse
import logging

osp = os.path

def create_plot_window(vis, xlabel, ylabel, title, win, env, trace_name):
  if not isinstance(trace_name, list):
    trace_name = [trace_name]

  vis.line(X=np.array([1]), Y=np.array([np.nan]), win=win, env=env,
    name=trace_name[0],
    opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))
  for name in trace_name[1:]:
    vis.line(X=np.array([1]), Y=np.array([np.nan]), win=win, env=env,
    name=name)


def train(data_dir, contactpose_dir, split, config_file, experiment_suffix=None,
    checkpoint_dir='.', device_id=0, weights_filename=None, resume_optim=False):
  model_name = config_file.split('/')[-1].split('.')[0]
  # config
  config = configparser.ConfigParser()
  config.read(config_file)

  section = config['optim']
  batch_size = section.getint('batch_size')
  max_epochs = section.getint('max_epochs')
  val_interval = section.getint('val_interval')
  do_val = val_interval > 0
  base_lr = section.getfloat('base_lr')
  momentum = section.getfloat('momentum')
  weight_decay = section.getfloat('weight_decay')

  section = config['misc']
  log_interval = section.getint('log_interval')
  shuffle = section.getboolean('shuffle')
  num_workers = section.getint('num_workers')
  visdom_server = section.get('visdom_server', 'http://localhost')

  section = config['hyperparams']
  droprate = section.getfloat('droprate')
  joint_droprate = section.getfloat('joint_droprate')
  lr_step_size = section.getint('lr_step_size', 10000)
  lr_gamma = section.getfloat('lr_gamma', 1.0)
  pos_weight = section.getfloat('pos_weight')
  n_rotations = section.getint('n_rotations')
  grid_size = section.getint('grid_size')
  uniform_texture_weights = section.getboolean('uniform_texture_weights', False)

  # cuda
  device = 'cuda:{:s}'.format(device_id)

  exp_name = '{:s}_split_{:s}_simple-joints'.format(model_name, split)
  if experiment_suffix:
    exp_name += '_{:s}'.format(experiment_suffix)
  checkpoint_dir = osp.join(checkpoint_dir, exp_name)
  
  # logging
  if not osp.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
  log_filename = osp.join(checkpoint_dir, 'training_log.txt')
  setup_logging(log_filename)
  logger = logging.getLogger()
  logger.info('Config from {:s}:'.format(config_file))
  with open(config_file, 'r') as f:
    for line in f:
      logger.info(line.strip())
  
  # create dataset, loss function and model
  texture_weights = np.load('data/texture_bin_weights.npy')
  texture_weights = torch.from_numpy(texture_weights).to(dtype=torch.float)
  if uniform_texture_weights:
    texture_weights = torch.ones(len(texture_weights))
  logger.info('Texture weights = %s' % texture_weights)
  splits = getattr(train_test_splits, 'split_{:s}'.format(split))
  n_surface_features = 3 + 21*3
  if 'pointnet' in model_name:
    model = PointNetPP(n_surface_features, len(texture_weights), droprate)
    loss_fn = tnn.CrossEntropyLoss(weight=texture_weights, ignore_index=-1)
    if do_val:
      val_loss_fn = tnn.CrossEntropyLoss(weight=texture_weights, ignore_index=-1)
  elif 'mlp' in model_name:
    n_hidden_nodes = config['hyperparams'].getint('n_hidden_nodes')
    model = MLPModel(n_surface_features, len(texture_weights), n_hidden_nodes,
        droprate)
    loss_fn = tnn.CrossEntropyLoss(weight=texture_weights, ignore_index=-1)
    if do_val:
      val_loss_fn = tnn.CrossEntropyLoss(weight=texture_weights, ignore_index=-1)
  else:
    raise NotImplementedError
  train_dset = ContactPose3D(data_dir, contactpose_dir, grid_size, n_rotations,
                             joint_droprate, **splits['train'])
  if do_val:
    val_dset = ContactPose3D(data_dir, contactpose_dir, grid_size, n_rotations, 0,
                              **splits['test'])
  # resume model
  if weights_filename is not None:
    checkpoint = torch.load(osp.expanduser(weights_filename))
    model.load_state_dict(checkpoint.state_dict(), strict=True)
    logger.info('Loaded weights from {:s}'.format(weights_filename))
  model.to(device=device)
  loss_fn.to(device=device)
  if do_val:
    val_loss_fn.to(device=device)

  # optimizer
  # optim = toptim.SGD(model.parameters(), lr=base_lr, weight_decay=weight_decay,
  #     momentum=momentum)
  optim = toptim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
  if isinstance(optim, toptim.Adam):
    lr_step_size = 1e10
    logger.info('Optimizer is Adam, disabling LR scheduler')
  lr_scheduler = toptim.lr_scheduler.StepLR(optim, step_size=lr_step_size,
      gamma=lr_gamma)
  
  # resume optim
  if (weights_filename is not None) and resume_optim:
    optim_filename = weights_filename.replace('model', 'optim')
    if osp.isfile(optim_filename):
      checkpoint = torch.load(optim_filename)
      optim.load_state_dict(checkpoint.state_dict())
      logger.info('Loaded optimizer from {:s}'.format(optim_filename))

  # dataloader
  train_dloader = DataLoader(train_dset, batch_size=batch_size, shuffle=shuffle,
    pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)
  if do_val:
    val_dloader = DataLoader(val_dset, batch_size=batch_size, shuffle=shuffle,
      pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)

  # checkpointing
  def checkpoint_fn(engine: Engine):
    return -engine.state.avg_loss

  checkpoint_kwargs = dict(dirname=checkpoint_dir, filename_prefix='checkpoint',
    score_function=checkpoint_fn, create_dir=True, require_empty=False,
      save_as_state_dict=False)
  checkpoint_dict = {'model': model, 'optim': optim}

  # train and val loops
  def train_loop(engine: Engine, batch):
    # occs, sdata, ijks, batch, colors = batch
    model.train()
    optim.zero_grad()
    
    if 'pointnet' in model_name:
      use_idx = range(1, len(batch))
      for idx in use_idx:
        batch[idx] = batch[idx].to(device=device, non_blocking=True)
      batch[2] = batch[2] / grid_size - 0.5
      pred = model(*batch[1:4])
      loss = loss_fn(pred, batch[4])
    elif 'mlp' in model_name:
      use_idx = [1, 4]
      for idx in use_idx:
        batch[idx] = batch[idx].to(device=device, non_blocking=True)
      pred = model(batch[1])
      loss = loss_fn(pred, batch[4])
    loss.backward()
    optim.step()
    engine.state.train_loss = loss.item()
    return loss.item()
  trainer = Engine(train_loop)
  train_checkpoint_handler = ModelCheckpoint(score_name='train_loss',
      **checkpoint_kwargs)
  trainer.add_event_handler(Events.EPOCH_COMPLETED, train_checkpoint_handler,
    checkpoint_dict)

  if do_val:
    def val_loop(engine: Engine, batch):
      # occs, sdata, ijks, batch, colors = batch
      model.eval()
      if 'pointnet' in model_name:
        use_idx = range(1, len(batch))
        for idx in use_idx:
          batch[idx] = batch[idx].to(device=device, non_blocking=True)
        batch[2] = batch[2] / grid_size - 0.5
        with torch.no_grad():
          pred = model(*batch[1:4])
          loss = val_loss_fn(pred, batch[4])
      elif 'mlp' in model_name:
        use_idx = [1, 4]
        for idx in use_idx:
          batch[idx] = batch[idx].to(device=device, non_blocking=True)
        with torch.no_grad():
          pred = model(batch[1])
          loss = val_loss_fn(pred, batch[4])
      engine.state.val_loss = loss.item()
      return loss.item()
    valer = Engine(val_loop)
    val_checkpoint_handler = ModelCheckpoint(score_name='val_loss',
        **checkpoint_kwargs)
    valer.add_event_handler(Events.EPOCH_COMPLETED, val_checkpoint_handler,
      checkpoint_dict)

  # callbacks
  vis = visdom.Visdom(server=visdom_server)
  logger.info('Visdom at {:s}'.format(visdom_server))
  loss_win = 'loss'
  create_plot_window(vis, '#Epochs', 'Loss', 'Training and Validation Loss',
    win=loss_win, env=exp_name, trace_name=['train_loss', 'val_loss'])

  @trainer.on(Events.ITERATION_COMPLETED)
  def log_training_loss(engine):
    it = (engine.state.iteration - 1) % len(train_dloader)
    engine.state.avg_loss = (engine.state.avg_loss*it + engine.state.output) / \
                            (it + 1)

    if it % log_interval == 0:
      logger.info("{:s} train Epoch[{:03d}/{:03d}] Iteration[{:04d}/{:04d}] "
          "Loss: {:02.4f} lr: {:.4f}".
        format(exp_name, engine.state.epoch, max_epochs, it+1, len(train_dloader),
        engine.state.output, lr_scheduler.get_lr()[0]))
      epoch = engine.state.epoch - 1 +\
              float(it+1)/len(train_dloader)

      vis.line(X=np.array([epoch]), Y=np.array([engine.state.output]),
        update='append', win=loss_win, env=exp_name, name='train_loss')

  @trainer.on(Events.EPOCH_COMPLETED)
  def log_avg_train_loss(engine):
    logger.info('{:s} Epoch[{:03d}/{:03d}] Avg. Training Loss: {:02.4f}'.format(
        exp_name, engine.state.epoch, max_epochs, engine.state.avg_loss))

  if do_val:
    @valer.on(Events.ITERATION_COMPLETED)
    def avg_loss_callback(engine: Engine):
      it = (engine.state.iteration - 1) % len(train_dloader)
      engine.state.avg_loss = (engine.state.avg_loss*it + engine.state.output) / \
                              (it + 1)
      if it % log_interval == 0:
        logger.info("{:s} val Iteration[{:04d}/{:04d}] Loss: {:02.4f}"
          .format(exp_name, it+1, len(val_dloader), engine.state.output))

    @valer.on(Events.EPOCH_COMPLETED)
    def log_val_loss(engine: Engine):
      vis.line(X=np.array([trainer.state.epoch]),
        Y=np.array([engine.state.avg_loss]), update='append', win=loss_win,
        env=exp_name, name='val_loss')

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_val(engine: Engine):
      vis.save([exp_name])
      if val_interval < 0:  # don't do validation
        return
      if engine.state.epoch % val_interval != 0:
        return
      valer.run(val_dloader)

    @trainer.on(Events.EPOCH_COMPLETED)
    def step_lr_scheduler(engine: Engine):
      lr_scheduler.step()

  def reset_avg_loss(engine: Engine):
    engine.state.avg_loss = 0
  trainer.add_event_handler(Events.EPOCH_STARTED, reset_avg_loss)
  if do_val:
    valer.add_event_handler(Events.EPOCH_STARTED, reset_avg_loss)

  # Ignite the torch!
  trainer.run(train_dloader, max_epochs)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir',
                      default=osp.join('data', 'simple-joints_prediction_data'))
  parser.add_argument(
      '--contactpose_dir', default=osp.join('data', 'contactpose_data'))
  parser.add_argument('--checkpoint_dir',
                      default=osp.join('data', 'checkpoints'))
  parser.add_argument('--split', type=str, required=True,
                      choices=('objects', 'participants', 'overfit'))
  parser.add_argument('--config_file', required=True)
  parser.add_argument('--weights_file', default=None)
  parser.add_argument('--suffix', default=None)
  parser.add_argument('--device_id', default='0')
  parser.add_argument('--resume_optim', action='store_true')
  args = parser.parse_args()

  train(args.data_dir, args.contactpose_dir, args.split, args.config_file,
    experiment_suffix=args.suffix, device_id=args.device_id,
    checkpoint_dir=osp.expanduser(args.checkpoint_dir),
    weights_filename=args.weights_file, resume_optim=args.resume_optim)
