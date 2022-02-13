import sys
import os
from pathlib import Path


CURR_FILE_PATH = (os.path.abspath(__file__))
PATH = Path(CURR_FILE_PATH)
CURR_DIR = str(PATH.parent.absolute())
sys.path.append(str(PATH.parent.parent.parent.absolute()))
sys.path.append(CURR_DIR)

import torch
from torchvision import transforms
from models.lit_models import Lit_Resnet_Transformer
# from models import *
from models.MLM_pretrain import MaskLanguageModel
from models.utils import get_best_checkpoint
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from data_preprocess.vocab import build_vocab,Vocabulary
from models.my_data_loader import CocoDataset
def decode(vocab: Vocabulary, pred: list):
    tokens = []
    for index in pred:
        token = vocab.idx2word[index]
        if token == '<end>':
            break
        if token == '<pad>' or token == '<start>':
            continue
        tokens.append(token)

    return tokens


def eval(myargs):
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    checkpoint = torch.load(os.path.join(CURR_DIR,"output/best_ckpt.pt"),map_location ='cpu')
    args = checkpoint['args']
    args.cuda_index = -1  #cpu
    vocab_txt_file_path = os.path.join(CURR_DIR,"resources/vocab.txt")
    vocab = build_vocab(vocab_txt_file_path)
    num_classes = len(vocab.word2idx)

    lit_model = lit_model = Lit_Resnet_Transformer(
                        args,
                        d_model=args.d_model, 
                        dim_feedforward=args.dim_feedforward,
                        nhead=args.nhead, 
                        dropout=args.dropout,
                        num_decoder_layers=args.num_decoder_layers,
                        max_output_len=args.max_output_len, 
                        sos_index=vocab('<start>'),  
                        eos_index=vocab('<end>'), 
                        pad_index=vocab('<pad>'),
                        unk_index=vocab('<unk>'), 
                        num_classes=num_classes,
                        lr=args.lr)
    lit_model.models.load_state_dict(checkpoint['model_state_dict'])
    with torch.no_grad():
        image = Image.open(os.path.join(CURR_DIR,"resources/evaluate_imgs/"+myargs.img_name))  
        image = CocoDataset.resize_image(image,size=256)
        image = transform(image)
        pred = lit_model.models.predict(image.unsqueeze(0).float())
        pred = pred.cpu().numpy()# type: ignore
        decoded = decode(vocab, pred[0].tolist())  # type: ignore
        decoded_str = " ".join(decoded)
        print(decoded_str)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_name", type=str,default="img1.png", 
                help="img which needs evaluation should be placed in 'Img2LaTeX/resources/evaluate_imgs/' ")

    myargs = parser.parse_args()
    eval(myargs)
