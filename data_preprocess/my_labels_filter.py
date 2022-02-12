# Created: 210313 14:02
# Last edited: 210421 14:02 

import os
import shutil
from tqdm import tqdm

def labels_filter(input_labels_dir_path, output_labels_no_error_dir_path, labels_num=None):
    if not (os.path.exists(output_labels_no_error_dir_path)):
        os.mkdir(output_labels_no_error_dir_path)
    labels_name_list = sorted(os.listdir(input_labels_dir_path))
    for i in tqdm(range(len(labels_name_list))):
        #when the labels_num is not None it means the labels_num is assigned
        if(labels_num is not None):
            if(i >= labels_num):
                break
        label_name = labels_name_list[i]
        #print(label_name)
        label_file_name = input_labels_dir_path +label_name
        with open(label_file_name, 'r', encoding='utf-8') as f1:
            content = f1.read()
        if 'error mathpix' in content:
            #print(content)
            continue
        shutil.copy(label_file_name, output_labels_no_error_dir_path + label_name)


'''
labels_filter_labels_num = 20
input_labels_dir_path = "data_preprocess/math_formula_images_grey_labels/"
output_labels_no_error_dir = 'data_preprocess/math_formula_grey_labels_no_error_mathpix/'
labels_filter(input_labels_dir_path=input_labels_dir_path, output_labels_no_error_dir_path=output_labels_no_error_dir, labels_num=labels_filter_labels_num)
'''


