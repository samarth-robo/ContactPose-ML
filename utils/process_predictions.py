import init_paths
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from eval import annealed_mean
import json

osp = os.path


def calc_auc(preds, targs, texture_bins, rebal_w, max_dist=1.0, N=500,
             plot_title=None, show=False):
  dists = np.abs(preds - targs)
  order = np.argsort(dists)
  preds = preds[order]
  targs = targs[order]
  dists = dists[order]
  
  threshes = np.linspace(0, max_dist, N+1)
  positives = np.zeros(len(threshes))
  rebal_positives = np.zeros(len(threshes))
  d_idx = 0
  t_idx = 1
  while True:
    tr_idx = max(0, np.searchsorted(texture_bins, targs[d_idx])-1)
    pr_idx = max(0, np.searchsorted(texture_bins, preds[d_idx])-1)
    # rw = max(rebal_w[tr_idx], rebal_w[pr_idx])
    rw = rebal_w[tr_idx]
    if dists[d_idx] <= threshes[t_idx]:
      positives[t_idx] += 1
      rebal_positives[t_idx] += rw
    else:
      t_idx += 1
      if t_idx == len(threshes):
        print('Difference {:f} > max threshold {:f}'.format(
            dists[d_idx], threshes[t_idx]))
        print('Please increase the range of thresholds to get accurate AuC')
        break
      positives[t_idx] = positives[t_idx-1]
      rebal_positives[t_idx] = rebal_positives[t_idx-1]
      d_idx -= 1
    d_idx += 1
    if d_idx == len(dists):
      break
  if t_idx < len(threshes)-1:
    positives[t_idx+1:] = positives[t_idx]
    rebal_positives[t_idx+1:] = rebal_positives[t_idx]
  positives = positives[1:] / positives[-1]
  rebal_positives = rebal_positives[1:] / rebal_positives[-1]
  threshes = threshes[1:]
  if show:
    plt.plot(threshes, positives, label='AuC')
    plt.plot(threshes, rebal_positives, label='re-balanced AuC')
    plt.xlabel('Texture Difference Threshold')
    plt.ylabel('Accuracy')
    plt.legend()
    if plot_title is not None:
      plt.title('{:s} AuC'.format(plot_title))
    plt.show()
  auc = np.sum(positives) / N
  rebal_auc = np.sum(rebal_positives) / N
  return auc, rebal_auc

def process(filename, annealed_mean_T=0.38, rebal_l=1, n_runs=3, show=False,
    no_rotations=False, direct_estimate=False):
  # use no_rotations=True for image-based experiments
  # in that case, axis=0 corresponds to different video frames
  # and we don't want to average over that
  texture_hists = np.load('data/texture_hists.npy')
  pbar = np.sum(texture_hists, axis=0).astype(float)
  pbar /= sum(pbar)
  rebal_w = rebal_l*pbar + (1-rebal_l)*np.ones(len(pbar)) / len(pbar)
  rebal_w = 1.0 / rebal_w
  rebal_w /= np.dot(rebal_w, pbar)
  texture_bins = np.linspace(0, 1, len(pbar)+1)
  texture_bin_centers = (texture_bins[:-1] + texture_bins[1:]) / 2.0
  
  id_idx = 0 if 'participants' in filename else 1
  output_dir, _ = osp.split(filename)
  filenames = [filename.replace(
      'run_1', 'run_{:d}'.format(r+1)) for r in range(n_runs)]
  aucs = {}
  rebal_aucs = {}
  rmses = {}
  for run_idx, filename in enumerate(filenames):
    print('Run {:d} / {:d}'.format(run_idx+1, n_runs))
    try:
      with open(filename, 'rb') as f:
        data = pickle.load(f)
    except Exception as e:
      print('Bad pickle {:s}'.format(filename))
      print(e)
      continue
    preds = {}
    targs = {}
    for this_data in data['data']:
      this_id = this_data[id_idx]
      pred, targ = this_data[-2:]
      # ignore the locations where we don't know the ground truth texture
      idx = targ >= 0
      targ = targ[idx]
      if no_rotations:
        pred = [p[idx] for p in pred]
        targ = [np.copy(targ) for _ in range(len(pred))]
        if this_id not in preds:
          preds[this_id] = pred
          targs[this_id] = targ
        else:
          preds[this_id].extend(pred)
          targs[this_id].extend(targ)
      else:
        pred = np.stack([p[idx] for p in pred]).mean(0)
        if this_id not in preds:
          preds[this_id] = [pred]
          targs[this_id] = [targ]
        else:
          preds[this_id].append(pred)
          targs[this_id].append(targ)

    for this_id in sorted(preds.keys()):
      p = []
      t = []
      if this_id not in aucs:
        aucs[this_id] = []
      if this_id not in rebal_aucs:
        rebal_aucs[this_id] = []
      if this_id not in rmses:
        rmses[this_id] = []
      for ps, ts in zip(preds[this_id], targs[this_id]):
        if not direct_estimate:
          # apply softmax
          ps = np.exp(ps) / np.sum(np.exp(ps), axis=1, keepdims=True)
          # make a prediction
          ps = annealed_mean(ps, texture_bins, T=annealed_mean_T)
        else:
          ps = np.squeeze(ps)
        p.append(ps)
        ts = texture_bin_centers[ts]
        t.append(ts)
      p = np.hstack(p)
      t = np.hstack(t)
      auc, rebal_auc = calc_auc(
          p, t, texture_bins, rebal_w, plot_title=this_id, show=show)
      rmse = np.linalg.norm(p-t) / np.sqrt(len(p))
      aucs[this_id].append(auc)
      rebal_aucs[this_id].append(rebal_auc)
      rmses[this_id].append(rmse)

  ids = []
  a = []
  a_stdev = []
  ra = []
  ra_stdev = []
  rmse = []
  for this_id in sorted(aucs.keys()):
    ids.append(this_id)
    a.append(np.mean(aucs[this_id]))
    a_stdev.append(np.std(aucs[this_id]))
    ra.append(np.mean(rebal_aucs[this_id]))
    ra_stdev.append(np.std(rebal_aucs[this_id]))
    rmse.append(np.mean(rmses[this_id]))
  s = 'identifiers: ' + ' '.join(['{:s}']*len(ids)).format(*ids)
  print(s)
  s = 'AUC: ' + ' '.join(['{:.4f}']*len(ids)).format(*a)
  print(s)
  s = 'AUC +/-: ' + ' '.join(['{:.4f}']*len(ids)).format(*a_stdev)
  print(s)
  s = 'Rebal AUC: ' + ' '.join(['{:.4f}']*len(ids)).format(*ra)
  print(s)
  s = 'Rebal AUC +/-: ' + ' '.join(['{:.4f}']*len(ids)).format(*ra_stdev)
  print(s)
  droprate = filename.split('=')[-1].split('_')[0]
  filename = osp.join(output_dir, 'results.json')
  try:
    with open(filename, 'r') as f:
      out = json.load(f)
  except FileNotFoundError:
    out = {}
  out[droprate] = {
      'identifiers': ids,
      'auc': a,
      'auc_stdev': a_stdev,
      'rebal_auc': ra,
      'rebal_auc_stdev': ra_stdev,
      'rmse': rmse}
  with open(filename, 'w') as f:
    json.dump(out, f, indent=4, separators=(',', ': '))
  print('{:s} written'.format(filename))

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--predictions_filename', required=True)
  parser.add_argument('--show', action='store_true')
  parser.add_argument('--n_runs', type=int, default=3)
  parser.add_argument('--no_rotations', action='store_true')
  parser.add_argument('--direct_estimate', action='store_true',
      help='We have direct estimates, do not run annealed mean')
  args = parser.parse_args()

  if args.direct_estimate:
    args.no_rotations = True
    print('Setting no_rotations to True')

  process(osp.expanduser(args.predictions_filename), n_runs=args.n_runs,
      show=args.show, no_rotations=args.no_rotations,
      direct_estimate=args.direct_estimate)