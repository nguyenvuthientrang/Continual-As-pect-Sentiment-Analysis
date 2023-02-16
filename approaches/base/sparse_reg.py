import sys,time
import numpy as np
import torch
import os
import logging
import glob
import math
import json
import argparse
import random
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data import TensorDataset, random_split
import lib_factory.utils as utils
import torch.nn.functional as F
import functools
import torch.nn as nn
from copy import deepcopy

def sparse_reg(outputs,targets,masks,mask_pre,lamb=0):
    """ masks and p_mask must have values in the same order """
    reg, count = 0., 0.

    if mask_pre is not None:
        for m, mp in zip(masks.values(), p_mask.values()):
            aux=1-mp
            reg+=(m*aux).sum()
            count+=aux.sum()
    else:
        for m in masks.values():
            reg+=m.sum()
            count+=np.prod(m.size()).item()

    reg/=count

    return lamb*reg

