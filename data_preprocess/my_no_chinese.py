import os
import shutil
from tqdm import tqdm
def FMM_func(user_dict, sentence):
    """
    正向最大匹配（FMM）
    :param user_dict: 词典
    :param sentence: 句子
    """
    # 词典中最长词长度
    max_len = max([len(item) for item in user_dict])
    start = 0
    token_list = []
    while start != len(sentence):
        index = start+max_len
        if index>len(sentence):
            index = len(sentence)
        for i in range(max_len):
            if (sentence[start:index] in user_dict) or (len(sentence[start:index])==1):
                token_list.append(sentence[start:index])
                # print(sentence[start:index], end='/')
                start = index
                break
            index += -1
    return token_list

def get_no_chinese_labels(input_vocab_file_path, input_labels_dir_path, 
                            output_labels_no_chinese_dir_path,labels_num, 
                            output_chinese_token_txt_file_path, output_labels_chinese_dir_path):

    if not (os.path.exists(output_labels_chinese_dir_path)):
        os.mkdir(output_labels_chinese_dir_path)
    if not (os.path.exists(output_labels_no_chinese_dir_path)):
        os.mkdir(output_labels_no_chinese_dir_path)


    labels_name_list = os.listdir(input_labels_dir_path)
    with open(input_vocab_file_path, 'r', encoding='utf-8') as f:
        vocab = f.read().split()

    max_token_len = 0
    for v in vocab:
        if len(v) > max_token_len:
            max_token_len = len(v)

    chinese_token_list=[]

    index = 1
    #this position is i0 because in FMM_func there is i
    for i0 in tqdm(range(len(labels_name_list))):
        if(labels_num is not None and i0 >= labels_num):
            break
        label_name = labels_name_list[i0]
        #print(index, ':')
        index += 1

        label_file_name = input_labels_dir_path + label_name
        with open(label_file_name, 'r', encoding='utf-8') as f1:
            content = f1.read()


        token_list = FMM_func(vocab, content)
        token_list = [token_list[i] for i in range(len(token_list)) if token_list[i] != ' '] # 去除空格

        new_content = ' '.join(token_list)

        have_chinese = False

        for token in token_list:
            if token not in vocab and token not in ['', ' ']:
                chinese_token_list.append(token)
                have_chinese = True

        if have_chinese is not True:
            # shutil.copy(label_file_name, output_label_dir + label_name)
            with open(output_labels_no_chinese_dir_path + label_name, 'w', encoding='utf-8') as f:
                f.write(new_content)
        else:
            with open(output_labels_chinese_dir_path + label_name, 'w', encoding='utf-8') as f:
                f.write(new_content)
    
    with open(output_chinese_token_txt_file_path, 'w', encoding='utf-8') as f:
        chinese_token_list = list(set(chinese_token_list))
        for chinese_token in chinese_token_list:
            f.write(chinese_token + '\n')



'''
labels_filter_labels_num = 100
input_labels_dir_path = 'data_preprocess/math_formula_images_grey_labels/'
output_labels_no_chinese_dir_path = 'data_preprocess/math_formula_images_grey_labels_no_chinese/'
input_vocab_file_path = 'data_preprocess/vocab.txt'
output_chinese_token_txt_file_path = 'data_preprocess/chinese_token.txt'
output_labels_chinese_dir_path = 'data_preprocess/math_formula_images_grey_labels_chinese/'
get_no_chinese_labels(input_vocab_file_path=input_vocab_file_path, 
                        labels_num=labels_filter_labels_num, input_labels_dir_path=input_labels_dir_path, 
                        output_labels_no_chinese_dir_path=output_labels_no_chinese_dir_path, 
                        output_chinese_token_txt_file_path=output_chinese_token_txt_file_path,
                        output_labels_chinese_dir_path=output_labels_chinese_dir_path)
'''

