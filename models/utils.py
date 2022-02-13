import os
import math

import torch
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
import numpy as np
import torch
import json
import tarfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
from urllib.request import urlretrieve
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from data_preprocess.my_build_vocab import *

def exist(file_path):
    return os.path.exists(file_path)

def get_checkpoint(ckpt_dir):
    """return full path if there is ckpt in ckpt_dir else None"""
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError("No checkpoint found in {}".format(ckpt_dir))

    ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith('ckpt')]
    
    if not ckpts:
        raise FileNotFoundError("No checkpoint found in {}".format(ckpt_dir))

    last_ckpt, max_epoch = None, 0
    for ckpt in ckpts:
        epoch = int(ckpt.split('-')[1])
        if epoch > max_epoch:
            max_epoch = epoch
            last_ckpt = ckpt
    full_path = os.path.join(ckpt_dir, last_ckpt)
    print("Get checkpoint from {} for training".format(full_path))
    return full_path


def get_best_checkpoint(ckpt_dir):
    """return full path if there is ckpt in ckpt_dir else None"""
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError("No checkpoint found in {}".format(ckpt_dir))

    ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith('best')]
    if not ckpts:
        raise FileNotFoundError("No checkpoint found in {}".format(ckpt_dir))

    last_ckpt, max_epoch = None, 0
    for ckpt in ckpts:
        last_ckpt = ckpt
    full_path = os.path.join(ckpt_dir, last_ckpt)
    print("Get checkpoint from {} for training".format(full_path))
    return full_path


# class EarlyStopping:
#     """Early stops the training if validation loss doesn't improve after a given patience."""
#     def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
#         """
#         Args:
#             patience (int): How long to wait after last time validation loss improved.
#                             Default: 7
#             verbose (bool): If True, prints a message for each validation loss improvement.
#                             Default: False
#             delta (float): Minimum change in the monitored quantity to qualify as an improvement.
#                             Default: 0
#             path (str): Path for the checkpoint to be saved to.
#                             Default: 'checkpoint.pt'
#             trace_func (function): trace print function.
#                             Default: print
#         """
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = np.Inf
#         self.delta = delta
#         self.path = path
#         self.trace_func = trace_func
#     def __call__(self, val_loss, model):

#         score = -val_loss

#         if self.best_score is None:
#             self.best_score = score
#             # self.save_checkpoint(val_loss, model)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             # self.save_checkpoint(val_loss, model)
#             self.counter = 0

#     # def save_checkpoint(self, val_loss, model):
#     #     '''Saves model when validation loss decrease.'''
#     #     if self.verbose:
#     #         self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
#     #     torch.save(model.state_dict(), self.path)
#     #     self.val_loss_min = val_loss



def decode(vocab: Vocabulary, pred: List[int]) -> List[str]:
    tokens = []
    for index in pred:
        token = vocab.idx2word[index]
        if token == '<end>':
            break
        if token == '<pad>' or token == '<start>':
            continue
        tokens.append(token)

    return tokens

def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize((size, size), Image.ANTIALIAS)


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
