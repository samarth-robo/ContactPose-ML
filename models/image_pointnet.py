import torch
import torch.nn as tnn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from models.pointnet import PointNetPP


def double_conv(in_channels, out_channels):
  return tnn.Sequential(
    tnn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
    tnn.ReLU(inplace=True),
    tnn.BatchNorm2d(out_channels),
    tnn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
    tnn.ReLU(inplace=True),
    tnn.BatchNorm2d(out_channels))


class EncoderDecoder(tnn.Module):
  def __init__(self, outplanes, inplanes=3):
    super(EncoderDecoder, self).__init__()

    self.dconv_down1 = double_conv(inplanes, 16)
    self.dconv_down2 = double_conv(16, 32)
    self.dconv_down3 = double_conv(32, 64)

    self.downsample = tnn.MaxPool2d(2)
    self.upsample   = tnn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    self.global_pool = tnn.AdaptiveAvgPool2d((1, 1))

    self.dconv_up2 = double_conv(64+64+32, 32)
    # self.dconv_up2 = double_conv(64+32, 32)
    self.dconv_up1 = double_conv(32+16, 16)
    
    self.conv_out = tnn.Conv2d(16, outplanes, 1)

    for m in self.modules():
      if isinstance(m, tnn.Conv2d):
        tnn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        tnn.init.constant_(m.bias, 0)
      elif isinstance(m, tnn.BatchNorm2d):
        tnn.init.constant_(m.weight, 1)
        tnn.init.constant_(m.bias, 0)

  def forward(self, x):
    conv1 = self.dconv_down1(x)
    x = self.downsample(conv1)

    conv2 = self.dconv_down2(x)
    x = self.downsample(conv2)

    x = self.dconv_down3(x)

    # comment this line
    xg = self.global_pool(x)

    x = self.upsample(x)
    x = torch.cat((x, conv2, xg.expand_as(x)), dim=1)
    # x = torch.cat((x, conv2), dim=1)

    x = self.dconv_up2(x)
    x = self.upsample(x)
    x = torch.cat((x, conv1), dim=1)

    x = self.dconv_up1(x)

    x = self.conv_out(x)
    return x


class ImagePointNet(tnn.Module):
  def __init__(self, num_features, num_classes, droprate, num_im_channels,
      num_cameras, num_views, training_mode=True):
    super(ImagePointNet, self).__init__()
    self.num_cameras = num_cameras
    self.num_views = num_views
    self.training_mode = training_mode
    self.camera_names = ['left', 'right', 'middle']
    assert((num_views >=1) and (num_views <= self.num_cameras))
    self.im_encdec = EncoderDecoder(num_features, num_im_channels)
    self.pointnet  = PointNetPP(num_features, num_classes, droprate,
        training_mode=self.training_mode)

  def forward(self, x_im, vert_info, pos, batch):
    x_im = x_im.view(-1, *x_im.shape[-3:])
    x_im = self.im_encdec(x_im)

    x = []
    # only used in eval mode to keep the view same across
    # rotations
    batch_camera = [torch.randperm(self.num_cameras)[0].item()]
    for batch_idx in range(len(vert_info)):
      feats = []
      indices = []
      if self.training_mode:
        cam_idxs = torch.randperm(self.num_cameras)[:self.num_views]
      else:
        cam_idxs = batch_camera * self.num_views
      for cam_idx in cam_idxs:
        # print('Camera {:s}'.format(self.camera_names[cam_idx]))
        vinfo = vert_info[batch_idx][cam_idx]
        im_idx = batch_idx*self.num_cameras + cam_idx
        feats.append(x_im[im_idx][:, vinfo[:, 1], vinfo[:, 0]])
        indices.append(torch.tensor(vinfo[:, 2], dtype=torch.long, device=x_im.device))
      feats = torch.cat(feats, 1)
      indices = torch.cat(indices)
      xb = scatter_mean(feats, indices, dim=1,
          dim_size=torch.sum(batch==batch_idx).item(), fill_value=0)
      x.append(xb)
    x = torch.cat(x, 1)

    out = self.pointnet(x.t(), pos, batch)
    return out
