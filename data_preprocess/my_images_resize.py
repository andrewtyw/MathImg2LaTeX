
import os
from PIL import Image
from torch import imag
from tqdm import tqdm
def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize((size, size), Image.ANTIALIAS)

def resize_images(input_images_dir_path, output_resized_images_dir_path, size):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_resized_images_dir_path):
        os.makedirs(output_resized_images_dir_path)

    images = os.listdir(input_images_dir_path)
    num_images = len(images)
    for i, image in tqdm(enumerate(images),total=len(images)):
        with open(os.path.join(input_images_dir_path, image), 'r+b') as f:
            with Image.open(f) as img:
                img = resize_image(img, size)
                img.save(os.path.join(output_resized_images_dir_path, image), img.format)
        # if (i+1) % 100 == 0:
        #     print ("[{}/{}] Resized the images and saved into '{}'."
        #            .format(i+1, num_images, output_resized_images_dir_path))


'''
input_images_dir_path = "data_preprocess/dataset/math_formula_images_grey_no_chinese/"
output_resized_images_dir_path = "data_preprocess/dataset/math_formula_images_grey_no_chinese_resized/"
resize_images(input_images_dir_path=input_images_dir_path, 
                    output_resized_images_dir_path=output_resized_images_dir_path, 
                            size=256)
'''


