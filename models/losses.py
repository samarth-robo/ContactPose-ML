import torch
import torch.nn as tnn


def average_rotated_pointcloud_predictions(preds, batch, n_batches):
  pred_list = []
  for batch_idx in range(n_batches):
    p = preds[batch == batch_idx]
    pred_list.append(p)
  avg_pred = torch.stack(pred_list).mean(0)
  return avg_pred

def average_rotated_grid_predictions(preds, ijks, batch):
  pred_list = []
  if ijks.dtype != torch.long:
    ijks = ijks.to(dtype=torch.long)
  for batch_idx in range(len(preds)):
    this_ijks = ijks[batch == batch_idx]
    p = preds[batch_idx, :, this_ijks[:, 2], this_ijks[:, 1], this_ijks[:, 0]]
    pred_list.append(p)
  avg_pred = torch.stack(pred_list).mean(0).t()
  return avg_pred


# only count the loss where target is non-zero
class TextureGridLoss(tnn.Module):
  def __init__(self, weight=None, eval_mode=False):
    super(TextureGridLoss, self).__init__()
    self.loss_fn = tnn.CrossEntropyLoss(weight=weight, ignore_index=-1)
    self.eval_mode = eval_mode

  def forward(self, pred, targ_ijks, targ_batch, targ_colors):
    targ_ijks = targ_ijks.to(dtype=torch.long)
    if self.eval_mode:
      pred = average_rotated_grid_predictions(pred, targ_ijks, targ_batch)
      targ_colors = targ_colors[targ_batch == 0]
    else:
      pred = pred[targ_batch, :, targ_ijks[:, 2],
                  targ_ijks[:, 1], targ_ijks[:, 0]]
    loss = self.loss_fn(pred, targ_colors)
    return loss
