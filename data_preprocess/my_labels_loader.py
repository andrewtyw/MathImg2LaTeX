import os

def get_labels_loader(labels_dir_path,labels_num):
    labels_content = []
    if not (os.path.exists(labels_dir_path)):
        print("the directory doesn't exist")
    labels_file_name = sorted(os.listdir(labels_dir_path))
    for i in range(len(labels_file_name)):
        if(i >= labels_num):
            break
        label_file_name = labels_file_name[i]
        label_file_path = labels_dir_path + '/' +label_file_name
        with open(label_file_path, 'r', encoding='utf-8') as f1:
            content = f1.read()
        labels_content.append(content)
        
    return labels_content

'''
labels_dir_path = "data_preprocess/math_formula_images_grey_labels"
labels_num = 10
labels_content = get_labels_loader(labels_num=labels_num, labels_dir_path=labels_dir_path)
print(labels_content)
print(len(labels_content))
'''

