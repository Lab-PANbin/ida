# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.SSDD import SSDD
from datasets.SAR import SAR
from datasets.LEVIR import LEVIR
from datasets.HRSC import HRSC

import numpy as np

for split in ['train', 'trainval','val','test']:
  name = 'SSDD_{}'.format(split)
  __sets[name] = (lambda split=split : SSDD(split))
for split in ['train', 'trainval','val','test']:
  name = 'SAR_{}'.format(split)
  __sets[name] = (lambda split=split : SAR(split))
for split in ['train', 'trainval','val','test']:
  name = 'DIOR_{}'.format(split)
  __sets[name] = (lambda split=split : DIOR(split))
for split in ['train', 'trainval','val','test']:
  name = 'LEVIR_{}'.format(split)
  __sets[name] = (lambda split=split : LEVIR(split))
for split in ['train', 'trainval','val','test']:
  name = 'HRSC_{}'.format(split)
  __sets[name] = (lambda split=split : HRSC(split))

def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
