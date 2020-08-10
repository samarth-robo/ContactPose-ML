"""
Downloads trained models and pre-processed data for contact modeling
"""
import os
import requests
from zipfile import ZipFile
from tqdm import tqdm

osp = os.path


def filename_from_response(r):
  filename = None
  for ss in r.headers['Content-Disposition'].split(';'):
    if not 'filename=' in ss:
      continue
    sidx, eidx = ss.find('"'), ss.rfind('"')
    filename = ss[sidx+1 : eidx]
  if filename is None:
    print('Response did not contain filename, something wrong')
  return filename


def download_url(url, dirname='.', filename=None, progress=True):
  """
  taken from https://stackoverflow.com/a/37573701
  """
  # Streaming, so we can iterate over the response.
  r = requests.get(url, stream=True)
  # Total size in bytes.
  total_size = int(r.headers.get('content-length', 0))
  block_size = 1024 #1 Kibibyte
  if progress:
    t=tqdm(total=total_size, unit='iB', unit_scale=True)

  # compose filename
  if filename is None:
    filename = filename_from_response(r)
  if filename is None:
    return filename
  filename = osp.join(osp.expanduser(dirname), filename)

  done = True
  datalen = 0
  print('Downloading {:s}...'.format(filename))
  with open(filename, 'wb') as f:
    itr = r.iter_content(block_size)
    while True:
      try:
        try:
          data = next(itr)
        except StopIteration:
          break
        if progress:
          t.update(len(data))
        datalen += len(data)
        f.write(data)
      except KeyboardInterrupt:
        done = False
        print('Cancelled')
  if progress:
    t.close()
  if (not done) or (total_size != 0 and datalen != total_size):
    print("ERROR, something went wrong")
    try:
      os.remove(filename)
    except OSError as e:
      print(e)
    return None
  else:
    return filename


def unzip_and_del(filename, dst_dir=None, progress=True):
  if dst_dir is None:
    dst_dir, _ = osp.split(filename)
  if len(dst_dir) == 0:
    dst_dir = '.'
  with ZipFile(filename) as f:
    f.extractall(dst_dir)
  os.remove(filename)


if __name__ == '__main__':
  import sys
  import random
  model_urls = [
    "https://www.dropbox.com/sh/jszbnc5txyp1lny/AABKT8Z9DeHlgZyP7hRxilToa?dl=1",
    "https://www.dropbox.com/sh/x6wl7pbj64y3zxa/AAASD_pFlaUVtgD6_lLIO2lQa?dl=1",
    "https://www.dropbox.com/sh/oo2xsjoklnxwfi7/AAB1By9ELXgpcxfxu11zvKFka?dl=1",
    "https://www.dropbox.com/sh/e6gm4mwsbaqml89/AACPnfzQtL1-YqoMpXkIiA4na?dl=1",
    "https://www.dropbox.com/sh/1ykqz8pddya1zu7/AABXAuwMIaBLncLmq2t_hoRIa?dl=1",
    "https://www.dropbox.com/sh/13mygnw3yu70f5u/AADIGFoV_HNBvoRc_iRGypURa?dl=1",
  ]
  binvoxes_url = \
    "https://www.dropbox.com/sh/zyy9jyo6pzat456/AABwO3cR6uVe0bKMXfXn55XQa?dl=1"
  prediction_data_url = \
    "https://www.dropbox.com/s/7xyrafply27efog/skeleton_prediction_data.zip?dl=1"

  ####################
  # Download trained models to data/checkpoints/<model_name>
  for url in model_urls:
    filename = download_url(url, dirname=osp.join('data', 'checkpoints'))
    if filename is None:
      continue
    dirname, model_name = osp.split(filename)
    model_name = model_name.replace('.zip', '')
    dirname = osp.join(dirname, model_name)
    if not osp.isdir(dirname):
      os.mkdir(dirname)
    unzip_and_del(filename, dst_dir=dirname)
    break
  
  ####################
  # Download object model voxelizations to data/binvoxes
  # binvoxes are common for all branches, so check if they have already been 
  # downloaded
  dirname = osp.join('data', 'binvoxes')
  if not osp.isdir(dirname):
    os.mkdir(dirname)
    filename = download_url(binvoxes_url, dirname='data')
    if filename is not None:
      unzip_and_del(filename, dst_dir=dirname)
  else:
    print('binvoxes have already been downloaded, skipping')
  
  ####################
  # Download pre-processed prediction data to data/<branch_name>_prediction_data
  filename = download_url(prediction_data_url, dirname='data')
  if filename is not None:
    _, dirname = osp.split(filename)
    dirname = dirname.replace('.zip', '')
    dirname = osp.join('data', dirname)
    if not osp.isdir(dirname):
      os.mkdir(dirname)
    unzip_and_del(filename, dst_dir=dirname)