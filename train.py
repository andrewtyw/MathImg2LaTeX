import sys
import os
from pathlib import Path


CURR_FILE_PATH = (os.path.abspath(__file__))
print(CURR_FILE_PATH)
PATH = Path(CURR_FILE_PATH)
CURR_DIR = str(PATH.parent.absolute())
sys.path.append(str(PATH.parent.parent.parent.absolute()))
sys.path.append(CURR_DIR)

import torch
from torchvision import transforms
from models.my_data_loader import get_loader
import torch
import pickle
from models.lit_models import Lit_Resnet_Transformer
from models.lit_models import Lit_Resnet_Transformer
import numpy as np
import nltk
import argparse
from models.utils import get_checkpoint,setup_seed,get_best_checkpoint
from models.training import Trainer
from models.MLM_pretrain import MaskLanguageModel
from data_preprocess.vocab import Vocabulary,build_vocab


def main(args):
    
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    # nltk.download('punkt')
    # get args
    print(args)
    setup_seed(args.seed) #设置随机种子
    max_epoch = args.epoches
    from_check_point = args.from_check_point
    if from_check_point:
        checkpoint_path = get_best_checkpoint("/data/zzengae/tywang/save_model/physics/from_MLM_pretrain_256")
        checkpoint = torch.load(checkpoint_path)
        # args = checkpoint['args']
    # print("Training args:", args)

    # nltk.download('punkt')


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3,1,1)),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))   # 这个可以改一下
    ])   # 修改的位置

    # labels_lst_file_path = "data/im2latex_formulas.norm.lst"
    # train_images_lst_file_path = "data/im2latex_train_filter.lst"
    # val_images_lst_file_path = "data/im2latex_validate_filter.lst"
    # images_dir_path = "data/math_formula_images_grey_no_chinese_resized/"

    # math
    # labels_lst_file_path = "/home/zzengae/WangTianyin/Latex_rc/scr_v8/full_data/im2latex_formulas.norm.lst"
    # train_images_lst_file_path = "/home/zzengae/WangTianyin/Latex_rc/scr_v8/full_data/im2latex_train_filter.lst"
    # val_images_lst_file_path = "/home/zzengae/WangTianyin/Latex_rc/scr_v8/full_data/im2latex_validate_filter.lst"
    # images_dir_path = "/home/zzengae/WangTianyin/Latex_rc/scr_v8/full_data/math_formula_images_grey_no_chinese_resized/"
    # train_batch_size = val_batch_size = 64
    # vocab_pkl_file_path = "/home/zzengae/WangTianyin/Latex_rc/scr_v8/full_data/vocab.pkl"

    labels_lst_file_path = '/data/tywang/img2latex/im2latex_formulas.norm.lst' 
    train_images_lst_file_path = "/data/tywang/img2latex/im2latex_train_filter.lst"
    val_images_lst_file_path = "/data/tywang/img2latex/im2latex_validate_filter.lst"
    images_dir_path = "/data/tywang/img2latex/math_formula_images_grey_no_chinese_resized/"
    
    # physics
    # labels_lst_file_path = "/data/zzengae/tywang/save_model/physics/im2latex_formulas.norm.lst"
    # train_images_lst_file_path = "/data/zzengae/tywang/save_model/physics/im2latex_train_filter.lst"
    # val_images_lst_file_path = "/data/zzengae/tywang/save_model/physics/im2latex_validate_filter.lst"
    # images_dir_path = "/data/zzengae/tywang/save_model/physics/physics_formula_images_grey_no_chinese_resized/"
    # train_batch_size = val_batch_size = 64
    # vocab_pkl_file_path = "/data/zzengae/tywang/save_model/physics/vocab.pkl"

    train_batch_size  = args.batch_size
    val_batch_size = 64

    vocab_txt_file_path = '/home/tywang/myURE/Img2LaTeX/resources/vocab.txt'
    vocab = build_vocab(vocab_txt_file_path)
    #根据bert-base-uncased的词表, [MASK] token 存在, 因此直接加[MASK] token 

    num_classes = len(vocab.word2idx)

    if args.MLM_pretrain_mode:
        MLM = MaskLanguageModel(vocab)
    else:MLM = None
    train_data_loader = get_loader(labels_lst_file_path= labels_lst_file_path, images_lst_file_path= train_images_lst_file_path,
                                images_dir_path= images_dir_path, batch_size= train_batch_size, vocab=vocab,
                                transform= transform)

    val_data_loader = get_loader(labels_lst_file_path= labels_lst_file_path, images_lst_file_path= val_images_lst_file_path,
                                images_dir_path= images_dir_path, batch_size= val_batch_size, vocab=vocab,
                                transform= transform)
    batch = next(iter(train_data_loader))
    lit_model = Lit_Resnet_Transformer(
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
                        num_classes=num_classes)
    print("init model ok")
    optimizer, scheduler = lit_model.configure_optimizers()

    if args.from_MLM:
        lit_model.models.load_state_dict(checkpoint['model_state_dict']) #MLM仅仅加载模型参数即可, 相当于重新训练
        trainer = Trainer(optimizer, 
                         lit_model, 
                         scheduler,
                         train_data_loader,  
                         val_data_loader, 
                         args,
                         MLM=MLM,
                         init_epoch=0, 
                         last_epoch=max_epoch)
    elif from_check_point:
        lit_model.models.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['lr_sche'])
        # init trainer from checkpoint
        trainer = Trainer(optimizer, 
                        lit_model, 
                        scheduler,
                        train_data_loader, 
                        val_data_loader, 
                        args,
                        MLM=MLM,
                        init_epoch=epoch, 
                        last_epoch=max_epoch)
    else:
        trainer = Trainer(optimizer, 
                         lit_model, 
                         scheduler,
                         train_data_loader,  
                         val_data_loader, 
                         args,
                         MLM=MLM,
                         init_epoch=0, 
                         last_epoch=max_epoch)
    # begin training
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Im2Latex Training Program")
    # model args
    # 训练数据文件夹
    parser.add_argument("--data_dir", type=str,
                        default="/data/zzengae/tywang/save_model/physics", # "/data/zzengae/tywang/save_model/math/math" 数学#############################################################
                        help="data_dir之下, 有这几个文件:"+
                        "im2latex_formulas.norm.lst"+
                        "im2latex_train_filter.lst"+
                        "im2latex_validate_filter.lst"+
                        "math_formula_images_grey_no_chinese_resized"+
                        "vocab.pkl"
                        )

    # Lit_Resnet_Transformer模型参数
    parser.add_argument("--max_output_len", type=int,
                        default=210, help="对于math,200合适, 150报错")
    parser.add_argument("--seed", type=int,
                        default=16, help="随机种子, 已确保模型能复现")
    parser.add_argument("--dropout", type=float,
                        default=0.2, help="transformer 的dropout")
    parser.add_argument("--d_model", type=int,
                        default=256, help="单词embedding的维度")
    parser.add_argument("--num_decoder_layers", type=int,
                        default=3, help="as named")
    parser.add_argument("--dim_feedforward", type=int,
                        default=256, help="as named")
    parser.add_argument("--nhead", type=int,
                        default=4, help="as named")
    #训练参数
    parser.add_argument("--batch_size", type=int, default=64) 
    parser.add_argument("--epoches", type=int, default=31)
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning Rate")
    parser.add_argument("--min_lr", type=float, default=3e-5,
                        help="Learning Rate")
    parser.add_argument("--sample_method", type=str, default="teacher_forcing",
                        choices=('teacher_forcing', 'exp', 'inv_sigmoid'),
                        help="The method to schedule sampling")
    parser.add_argument("--decay_k", type=float, default=1.,
                        help="Base of Exponential decay for Schedule Sampling. "
                             "When sample method is Exponential deca;"
                             "Or a constant in Inverse sigmoid decay Equation. "
                             "See details in https://arxiv.org/pdf/1506.03099.pdf"
                        )
    parser.add_argument("--lr_decay", type=float, default=0.5,
                        help="Learning Rate Decay Rate")
    parser.add_argument("--lr_patience", type=int, default=3,
                        help="Learning Rate Decay Patience")
    parser.add_argument("--clip", type=float, default=2.0,
                        help="The max gradient norm")
    
    parser.add_argument("--print_freq", type=int, default=1,
                        help="The frequency to print message")

    parser.add_argument("--from_check_point", action='store_true',
                        default=False, help="Training from checkpoint or not")
    parser.add_argument("--accumulation_step", default=1, help="梯度累计步数") 

    #训练模式
    parser.add_argument("--MLM_pretrain_mode",type=bool, default=False, help="是否MLM预训练模式")
    parser.add_argument("--from_MLM",type=bool, default=False, help="是否使用预训练模型进行训练")
    parser.add_argument("--test_mode", default=False, help="test mode=true, train& validate会只会分别跑3个step,用于检查代码有没有bug")

    #设备相关
    parser.add_argument("--cuda_index", default=3, help="cuda设备index")

    #文件保存相关
    parser.add_argument("--save_dir", type=str,
                        default="/data/tywang/img2latex/save_model",    
                        help="checkpoints, best-model, 评价指标的变化图, 评价指标数据,都会存到这个文件夹之下")
    
    args = parser.parse_args()


    main(args)
