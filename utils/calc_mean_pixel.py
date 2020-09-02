from datasets.contact_pose_images import ContactPoseImages, rgb_collate_fn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from train_test_splits import split_images_objects


def calc_mean_pixel():
  dset = ContactPoseImages('data/images_prediction_data',
                           'data/contactpose_data', 64, 1, use_depth=False,
                           rgb_transform=transforms.ToTensor(),
                           **split_images_objects['train'])
  dloader = DataLoader(dset, batch_size=30, collate_fn=rgb_collate_fn,
                       shuffle=True)

  px = np.zeros(3)
  px_sq = np.zeros(3)
  N = 0
  for batch_idx, batch in enumerate(dloader):
    if batch_idx % 50 == 0:
      print('Batch {:d} / {:d}'.format(batch_idx, len(dloader)))

    rgb_images = batch[0]
    rgb_images = rgb_images.permute(0, 1, 3, 4, 2).reshape(-1, 3).numpy()

    px += np.sum(rgb_images, axis=0)
    px_sq += np.sum(rgb_images**2, axis=0)
    N += len(rgb_images)

    if batch_idx > 10:
      break

  mean_px = px / N
  var_px  = (px_sq / N) - mean_px**2
  std_px  = np.sqrt(var_px)
  print('mean = ', mean_px)
  print('std = ', std_px)


if __name__ == '__main__':
  calc_mean_pixel()
