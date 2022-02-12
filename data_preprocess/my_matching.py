#the function of this module is to make the correct dir according to label file name
import os
import shutil
from tqdm import tqdm

def matching_labels_images_dir(input_labels_dir_path, input_images_dir_path, output_images_dir_path):
    if not (os.path.exists(output_images_dir_path)):
        os.mkdir(output_images_dir_path)

    labels_name_list = os.listdir(input_labels_dir_path)

    for label_file_name in tqdm(labels_name_list):
        image_file_name = label_file_name[:-4] + '.png'
        ouput_image_file_path = output_images_dir_path + image_file_name
        input_image_file_path = input_images_dir_path + image_file_name
        if(os.path.exists( input_image_file_path)):
            shutil.copy(  input_image_file_path ,output_images_dir_path + image_file_name)
        else:
            print(image_file_name, "doesn't exist")


'''
input_labels_dir_path = 'data_preprocess/math_formula_images_grey_labels_no_chinese/'
input_images_dir_path = 'data_preprocess/math_formula_images_grey/'
output_images_dir_path = 'data_preprocess/math_formula_images_grey_no_chinese/'
matching_labels_images_dir(input_labels_dir_path=input_labels_dir_path, input_images_dir_path=input_images_dir_path,
                            output_images_dir_path=output_images_dir_path)
'''




