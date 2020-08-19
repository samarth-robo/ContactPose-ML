import torch
import torch.nn as tnn


# only count the loss where target is non-zero
class TextureGridLoss(tnn.Module):
  def __init__(self, weight=None, eval_mode=False):
    super(TextureGridLoss, self).__init__()
    self.loss_fn = tnn.CrossEntropyLoss(weight=weight, ignore_index=-1)
    self.eval_mode = eval_mode

  def forward(self, pred, targ_ijks, targ_batch, targ_colors):
    targ_ijks = targ_ijks.to(dtype=torch.long)
    if self.eval_mode:
      pred_list = []
      for b in range(len(pred)):
        this_ijks = targ_ijks[targ_batch == b]
        pred_list.append(pred[b, :, this_ijks[:, 2],
                          this_ijks[:, 1], this_ijks[:, 0]])
      pred = torch.stack(pred_list).mean(0)
      targ_colors = targ_colors[targ_batch == 0]
    else:
      pred = pred[targ_batch, :, targ_ijks[:, 2],
                  targ_ijks[:, 1], targ_ijks[:, 0]]
    loss = self.loss_fn(pred, targ_colors)
    return loss
