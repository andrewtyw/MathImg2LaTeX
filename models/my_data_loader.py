import sys
import os
from pathlib import Path


CURR_FILE_PATH = (os.path.abspath(__file__))
print(CURR_FILE_PATH)
PATH = Path(CURR_FILE_PATH)
CURR_DIR = str(PATH.parent.absolute())
sys.path.append(str(PATH.parent.parent.absolute()))
sys.path.append(CURR_DIR)

import linecache
import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
import torch.utils.data as data
import nltk
from data_preprocess.vocab import Vocabulary,build_vocab
import pickle


class CocoDataset(data.Dataset):
    def __init__(self, labels_lst_file_path, 
    images_lst_file_path, images_dir_path, vocab,transform = None,size=256) -> None:
        super().__init__()
        self.images_path = []
        self.labels = []
        self.transform = transform
        self.vocab = vocab
        self.size = size

        index_list = []
        with open(images_lst_file_path, 'r') as f:  # eg '/data/tywang/img2latex/im2latex_train_filter.lst'
            for line in f:
                image_name = line[:-9]  # 这个匹配写死了, 不要  优化
                self.images_path.append(os.path.join(images_dir_path , image_name))  
                index = int(line[-8:-1])
                index_list.append(index)

            for index in index_list:
                #here we read until -1 because we want to ignore the /n
                label = linecache.getline(labels_lst_file_path, index+1)[:-1]
                self.labels.append(label)
    def resize_image(self,image, size):
        """Resize an image to the given size."""
        return image.resize((size, size), Image.ANTIALIAS)
    def __getitem__(self, index):
        image_path = self.images_path[index]
        pil_image = Image.open(image_path)
        pil_image = self.resize_image(pil_image, self.size)
        label = self.labels[index]
        if self.transform:
            image = self.transform(pil_image)
        else:
            pil_image = np.asarray(pil_image)
            image = torch.from_numpy(pil_image)
        tokens = nltk.tokenize.word_tokenize(label)
        label_ids = []
        label_ids.append(self.vocab('<start>'))
        label_ids.extend([self.vocab(token) for token in tokens])
        label_ids.append(self.vocab('<end>'))
        target = torch.Tensor(label_ids)
        #print('test target', target.shape)
        #print(target)
        return image, target  # 这里可以优化一下, 每取一次数据就要transform一次

    def __len__(self):
        return len(self.labels)

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, labels_ids = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(label_ids) for label_ids in labels_ids]
    targets = torch.zeros(len(labels_ids), max(lengths)).long()
    for i, label_ids in enumerate(labels_ids):
        end = lengths[i]
        targets[i, :end] = label_ids[:end]        
    return images, targets, lengths

def get_loader(labels_lst_file_path, images_lst_file_path, images_dir_path, vocab,batch_size, transform = None):

    coco = CocoDataset(labels_lst_file_path=labels_lst_file_path, images_lst_file_path=images_lst_file_path,
                        images_dir_path=images_dir_path, vocab=vocab,transform=transform)
                        
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=False,
                                              collate_fn=collate_fn)
    return data_loader


'''
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
        # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 标准化至[-1,1]
    ])
    labels_lst_file_path = "data_preprocess/dataset/im2latex_formulas.norm.lst"
    images_lst_file_path = "data_preprocess/dataset/im2latex_train_filter.lst"
    images_dir_path = "data_preprocess/dataset/math_formula_images_grey_no_chinese_resized/"
    vocab_pkl_file_path = "data_preprocess/dataset/vocab.pkl"
    batch_size = 5


    data_loader = get_loader(labels_lst_file_path, images_lst_file_path,
                                    images_dir_path, vocab_pkl_file_path,batch_size, transform)


    i = 0
    for i, (images, labels_ids, length) in enumerate(data_loader):
        if(i >= 2):
            break
        print(images.shape)
        #print(images[0].shape)
        print(images[0])
        print(labels_ids)
        #print(len(labels))
        #print(labels)
        i += 1
'''

