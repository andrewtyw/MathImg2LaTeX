# Created: 210313 14:02
# Last edited: 210421 14:02

import os
from nltk.tokenize import word_tokenize

def makeTxt(input_labels_dir_path,txt_path,labels_num=None):
    labels_name_list = sorted(os.listdir(input_labels_dir_path))
    for i in range(len(labels_name_list)):
        #when the labels_num is not None it means the labels_num is assigned
        if(labels_num is not None):
            if(i >= labels_num):
                break
        label_name = labels_name_list[i]
        #print(label_name)
        label_file_name = input_labels_dir_path +label_name
        with open(label_file_name, 'r', encoding='utf-8') as f1:
            text = f1.read()
            tokens = word_tokenize(text)
            print("tokens: {}".format(tokens))
            for token in tokens:
                inTxt(txt_path=txt_path, token=token)

def inTxt(txt_path, token):
    inflag = False
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if token == line[:-1]:
                inflag = True
                break
        if inflag == False:
            print("extra token: {}".format(token))




txt_path = "../data/physic_vocab.txt"
labels_filter_labels_num = 1000
input_labels_dir_path = "../data/physics_formula_images_grey_labels/"
makeTxt(input_labels_dir_path=input_labels_dir_path, txt_path=txt_path, labels_num=labels_filter_labels_num)



