import sys
import os
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils import data
from models.positional_encoding import PositionalEncoding1D, PositionalEncoding2D
import torchvision
from torchvision import transforms
from models.my_data_loader import get_loader
from data_preprocess.my_build_vocab import Vocabulary
import data_preprocess.my_build_vocab as my_build_vocab
import torch
import math
import pickle
from models import *
from models.lit_models import Lit_Resnet_Transformer
import numpy as np
import matplotlib.pyplot as plt
from models.lit_models import Lit_Resnet_Transformer
import numpy as np
import matplotlib.pyplot as plt
import nltk
import argparse
from models.utils import get_checkpoint,setup_seed

class MaskLanguageModel:
    def __init__(self,vocab:Vocabulary) -> None:
        vocab.add_word('[MASK]')
        self.MASK_id = vocab.word2idx['[MASK]']
        self.vocab = vocab
    def select_random_words(self,L:int):
        """
        在vocab 中随机选出L个单词的ids
        """
        can_be_select = list(self.vocab.idx2word.keys())[4:-1]
        can_be_select = np.array(can_be_select)
        random_selected = np.random.choice(can_be_select,(L,))
        return torch.from_numpy(random_selected).long()
    def generate_batch_mask_ids(self,batch):
        """
        根据trainloader的某个batch得到相应的mask_ids
        """
        _,labels_ids,_ = batch
        mask_label_ids = labels_ids.clone()
        for i,(ids,selected) in enumerate(zip(mask_label_ids,mask_label_ids!=0)):
            label = ids[selected]               #取出除了padding以外的单词
            L = torch.sum(selected)             # 长度
            rand = torch.rand(label.shape)      #产生随机数矩阵
            rand[0] = rand[-1] = 1              #保证开始符号和结束符号不会被mask掉
            # 替换0.15*0.8的部分为[MASK]    
            label[rand<0.12] = self.MASK_id     # <0.15中的80%替换成'[MASK]'
            rand[rand<0.12] = 1                 # 保证不会再被重复选择
            
            # 0.15*0.1的部分替换成别的词
            replace = rand<(0.15*0.9)
            L = torch.sum(replace)              #得到有多少个需要被替换成别的词
            label[replace] = self.select_random_words(L)
            mask_label_ids[i][selected] = label
        return labels_ids,mask_label_ids

            
    
    

