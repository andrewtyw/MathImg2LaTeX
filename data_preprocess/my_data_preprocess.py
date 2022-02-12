import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image


from my_labels_filter import *
from my_no_chinese import *
from my_im2latex import *
from my_matching import *
from my_images_resize import *
from my_build_vocab import Vocabulary,build_vocab
import argparse


def data_process():
    # labels_filter_labels_num = None
    # input_labels_dir_path = "/data/tywang/img2latex/math_formula_images_grey_labels/"
    # output_labels_no_error_dir = '/data/tywang/img2latex/math_formula_images_grey_labels_no_error_mathpix/'
    # labels_filter(input_labels_dir_path=input_labels_dir_path, output_labels_no_error_dir_path=output_labels_no_error_dir, labels_num=labels_filter_labels_num)
    # # 首先把label含有'error mathpix' 的数据给丢掉


    # get_no_chinese_labels_num = None
    # input_labels_dir_path = output_labels_no_error_dir
    # output_labels_no_chinese_dir_path = '/data/tywang/img2latex/math_formula_images_grey_labels_no_chinese/'
    # input_vocab_file_path = '/home/tywang/myURE/Img2LaTeX/resources/vocab.txt'
    # output_chinese_token_txt_file_path = '/home/tywang/myURE/Img2LaTeX/resources/chinese_token.txt'
    # output_labels_chinese_dir_path = '/data/tywang/img2latex/math_formula_images_grey_labels_grey_labels_chinese/'
    # get_no_chinese_labels(input_vocab_file_path=input_vocab_file_path,
    #                         labels_num=get_no_chinese_labels_num, input_labels_dir_path=input_labels_dir_path,
    #                         output_labels_no_chinese_dir_path=output_labels_no_chinese_dir_path,
    #                         output_chinese_token_txt_file_path=output_chinese_token_txt_file_path,
    #                         output_labels_chinese_dir_path=output_labels_chinese_dir_path)

    # input_matching_labels_dir_path = output_labels_no_chinese_dir_path
    # input_matching_images_dir_path = '/data/tywang/img2latex/math_formula_images_grey/'
    output_matching_images_dir_path = '/data/tywang/img2latex/math_formula_images_grey_no_chinese/'
    # matching_labels_images_dir(input_labels_dir_path=input_matching_labels_dir_path, input_images_dir_path=input_matching_images_dir_path,
    #                             output_images_dir_path=output_matching_images_dir_path)



    # input_labels_dir_path = '/data/tywang/img2latex/math_formula_images_grey_labels_no_chinese/'  # 对应的label的目录
    # output_lst_file_path = '/data/tywang/img2latex/im2latex_formulas.norm.lst'   # 输出的目录
    # output_lst_dir_path = '/data/tywang/img2latex/'
    # input_images_file_path = '/data/tywang/img2latex/math_formula_images_grey_no_chinese/'  # 对应的image的目录
    # labels_num = None
    # split_to_train_val_test(input_labels_dir_path=input_labels_dir_path, output_lst_dir_path=output_lst_dir_path,
    #                             output_lst_file_path=output_lst_file_path, input_images_file_path=input_images_file_path,
    #                                 labels_num=labels_num)


    input_images_dir_path = output_matching_images_dir_path
    output_resized_images_dir_path = "/data/tywang/img2latex/math_formula_images_grey_no_chinese_resized/"
    resize_images(input_images_dir_path=input_images_dir_path,
                        output_resized_images_dir_path=output_resized_images_dir_path,
                                size=256)

    vocab_txt_file_path = '/home/tywang/myURE/Img2LaTeX/resources/vocab.txt'
    vocab = build_vocab(vocab_txt_file_path)
    vocab_pkl_file_path = '/home/tywang/myURE/Img2LaTeX/resources/vocab.pkl'
    with open(vocab_pkl_file_path, 'wb') as f:
        pickle.dump(vocab, f)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Im2Latex Training Program")
    # parser.add_argument("--input_labels_dir_path", type=str,default="/data/zzengae/tywang/save_model/physics/physics_formula_images_grey_labels/")
    # parser.add_argument("--output_labels_no_error_dir", type=str,
    # default="")
    # parser.add_argument("--output_labels_no_chinese_dir_path", type=str,default="")
    # parser.add_argument("--input_vocab_file_path", type=str,default="")
    # parser.add_argument("--output_chinese_token_txt_file_path", type=str,default="")
    # parser.add_argument("--output_labels_chinese_dir_path", type=str,default="")
    # parser.add_argument("--input_matching_labels_dir_path", type=str,default="")
    # parser.add_argument("--input_matching_images_dir_path", type=str,default="")
    # parser.add_argument("--output_matching_images_dir_path", type=str,default="")
    # parser.add_argument("--output_lst_file_path", type=str,default="")
    # parser.add_argument("--output_lst_dir_path", type=str,default="")
    # parser.add_argument("--input_images_dir_path", type=str,default="")
    # parser.add_argument("--output_resized_images_dir_path", type=str,default="")
    # parser.add_argument("--vocab_txt_file_path", type=str,default="")
    # parser.add_argument("--vocab_pkl_file_path", type=str,default="")
    # parser.add_argument("--", type=str,default="")
    # args = parser.parse_args()
    data_process()