import os
import random
import shutil
from tqdm import tqdm
def split_to_train_val_test(input_labels_dir_path, output_lst_dir_path, output_lst_file_path,
                        input_images_file_path, labels_num=None):
                        
    if not (os.path.exists(output_lst_dir_path)):
        os.mkdir(output_lst_dir_path)
    # shuffle
    # image_output_dir = '../data/math_210421/formula_images/'

    #when the images_num is assigned it means we just want part of the dataset.
    if(labels_num is not None):
        images_name_list = sorted(os.listdir(input_images_file_path))[:labels_num]
        label_name_list = sorted(os.listdir(input_labels_dir_path))[:labels_num]
    #when the images_num is not assigned it means we want the whole dataset.
    else:  # 排序就可以让txt和png的名字对得上
        images_name_list = sorted(os.listdir(input_images_file_path))
        label_name_list = sorted(os.listdir(input_labels_dir_path))
    #print(images_name_list) #cuddly test
    #print(label_name_list) #cuddly test
    random.shuffle(label_name_list)   # 打乱label
    total_num = len(label_name_list)
    # train:val:test = 7:1:2
    train_num = int(0.7 * total_num)
    val_num = int(0.1 * total_num)
    test_num = total_num - train_num - val_num
    # 接下来三个list分别放 split出来的xxx.txt (label)
    train_list = []
    val_list = []
    test_list = []

    for i in range(total_num):
        if i < train_num:
            train_list.append(label_name_list[i])
        elif i < train_num + val_num:
            val_list.append(label_name_list[i])
        else:
            test_list.append(label_name_list[i])

    # output_lst_file_path = '/data/tywang/img2latex/im2latex_formulas.norm.lst'
    with open(output_lst_file_path, 'w', encoding='utf-8') as f0:  # f0是用来写入所有的label的
        index = 0
        with open(output_lst_dir_path + 'im2latex_train_filter.lst', 'w', encoding='utf-8') as f1:  #f1用于写入img(分开了train, dev, test)来写入
            for i in tqdm(range(train_num)):
                #print(index, end='\r')
                train_label_name = train_list[i]
                #image_name = train_label_name[:-4] + '.png' 
                image_name = train_label_name.split(".txt")[0]+'.png' 
                if image_name in images_name_list:
                    f1.write(image_name + ' ' + str(index).zfill(7) + '\n')
                    # shutil.copy(image_input_dir + image_name, image_output_dir + 'images_train/' + str(i) + '.png')
                    with open(input_labels_dir_path + train_label_name, 'r', encoding='utf-8') as f2:
                        line = f2.read()  # '\\left[ \\pi , \\frac { 4 \\pi } { 3 } \\right]' latex公式
                        f0.write(line + '\n')
                    index += 1

        with open(output_lst_dir_path + 'im2latex_validate_filter.lst', 'w', encoding='utf-8') as f1:
            for i in tqdm(range(val_num)):
                #print(index, end='\r')
                val_label_name = val_list[i]
                image_name = val_label_name[:-4] + '.png'
                if image_name in images_name_list:
                    f1.write(image_name + ' ' + str(index).zfill(7) + '\n')
                    # shutil.copy(image_input_dir + image_name, image_output_dir + 'images_val/' + str(i) + '.png')
                    with open(input_labels_dir_path + val_label_name, 'r', encoding='utf-8') as f2:
                        line = f2.read()
                        f0.write(line + '\n')
                    index += 1

        with open(output_lst_dir_path + 'im2latex_test_filter.lst', 'w', encoding='utf-8') as f1:
            for i in tqdm(range(test_num)):
                #print(index, end='\r')
                test_label_name = test_list[i]
                image_name = test_label_name[:-4] + '.png'
                if image_name in images_name_list:
                    f1.write(image_name + ' ' + str(index).zfill(7) + '\n')
                    # shutil.copy(image_input_dir + image_name, image_output_dir + 'images_test/' + str(i) + '.png')
                    with open(input_labels_dir_path + test_label_name, 'r', encoding='utf-8') as f2:
                        line = f2.read()
                        f0.write(line + '\n')
                    index += 1
    
    

'''
input_labels_dir_path = 'data_preprocess/math_formula_images_grey_labels/'
output_lst_file_path = 'data_preprocess/im2latex_formulas.norm.lst'
output_lst_dir_path = 'data_preprocess/'
input_images_file_path = 'data_preprocess/math_formula_images_grey'
images_num = 20
split_to_train_val_test(input_labels_dir_path=input_labels_dir_path, output_lst_dir_path=output_lst_dir_path,
                            output_lst_file_path=output_lst_file_path, input_images_file_path=input_images_file_path,
                                images_num=images_num)
'''

