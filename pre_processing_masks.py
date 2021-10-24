### PRE PROCESSING FUNCTIONS FOR DILATING GROUND TRUTH

import pathlib
import cv2
import os
import numpy as np
from os import listdir
from os.path import isfile, join

from PIL import Image


def dilate_mask(mask):
    kernel = np.ones((4,4),np.uint8)
    mask = cv2.dilate(mask,kernel,iterations = 1)
    mask_out = Image.fromarray(mask)
    return mask_out


def save_mask(mask_out, file_name):
    try:
        os.mkdir('./masks/')
    except:
        pass

    mask_filename = f'./masks/{file_name}.PNG'
    mask_out.save(mask_filename)
    

def save_original_as_png(image_folder, image_file):
    try:
        os.mkdir('./images/')
    except:
        pass
    stem = pathlib.Path(image_file).stem
    img = cv2.imread(f"{image_folder}/{image_file}")
    img_filename = f'./images/{stem}.PNG'
    cv2.imwrite(img_filename, img) 


def create_ground_truth(image_folder, fractures_folder):
    image_files = [f for f in listdir(image_folder) if isfile(join(image_folder, f))]
    print(image_files)
    for image_file in image_files:
        img = cv2.imread(f"{image_folder}/{image_file}", 0)
        non_black = cv2.countNonZero(img)
        if non_black == 0 or img.shape != (500, 500):
            continue

        mask = cv2.imread(f'{fractures_folder}/{image_file}',0)
        fractures_pixels = cv2.countNonZero(mask)
        if fractures_pixels == 0:
            continue
            
        mask_out = dilate_mask(mask)
        file_name = pathlib.Path(image_file).stem
        save_mask(mask_out, file_name)
        save_original_as_png(image_folder, image_file)
    return True

