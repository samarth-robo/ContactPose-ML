import logging
import sys
from datetime import datetime
import os
osp = os.path

def setup_logging(filename=None, noinfo=False):
  handlers = []
  handlers.append(logging.StreamHandler(sys.stdout))
  if filename is not None:
    time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    front, back = osp.split(filename)
    name, ext = back.split('.')
    back = '{:s}_{:s}.{:s}'.format(name, time, ext)
    filename = osp.join(front, back)
    handlers.append(logging.FileHandler(filename, 'w'))

  level = logging.WARN if noinfo else logging.INFO  
  logging.basicConfig(level=level, handlers=handlers)

  if filename is not None:
    root = logging.getLogger()
    root.info('Logging to {:s}'.format(filename))