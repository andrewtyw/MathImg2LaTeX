import sys
import os
from pathlib import Path
import argparse

CURR_FILE_PATH = (os.path.abspath(__file__))
print(CURR_FILE_PATH)
PATH = Path(CURR_FILE_PATH)
CURR_DIR = str(PATH.parent.absolute())
sys.path.append(CURR_DIR)

from data_preprocess.split import split_to_train_val_test

def prepare_main(args):
    """
        the dataset should be in 'Img2LaTeX/resources/data/'
            'Img2LaTeX/resources/data/imgs' contains all the LaTeX images
            'Img2LaTeX/resources/data/labels' contains the corresponding label
            the file name of images and labels must be the same  e.g.  abc.png->abc.txt
    """
    input_labels_dir_path = os.path.join(CURR_DIR,"resources/data/labels")   # 对应的label的目录
    input_images_file_path = os.path.join(CURR_DIR,"resources/data/imgs")  # 对应的image的目录
    output_lst_file_path = os.path.join(CURR_DIR,"resources/data/im2latex_formulas.norm.lst")  # 输出的目录
    output_lst_dir_path = os.path.join(CURR_DIR,"resources/data/") # 输出的目录
    split_to_train_val_test(args,input_labels_dir_path=input_labels_dir_path, output_lst_dir_path=output_lst_dir_path,
                                output_lst_file_path=output_lst_file_path, input_images_file_path=input_images_file_path,
                                    labels_num=None)

    print("data preparation done!")



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_val_test_ratio", type=str,default="7:2:1", help="as named")
    args = parser.parse_args()
    prepare_main(args)