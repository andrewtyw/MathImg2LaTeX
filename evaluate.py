import os
from os.path import join

import torch
from tqdm.std import tqdm
from models.lit_models import Lit_Resnet_Transformer
# from models import *
from models.MLM_pretrain import MaskLanguageModel
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

def eval(args):
    pass

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_ckpt_args_path", type=str,default="", help="as named")
    parser.add_argument("--checkpoint_path", type=str,default="", help="as named")
    args = parser.parse_args()
