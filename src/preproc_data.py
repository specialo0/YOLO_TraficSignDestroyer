import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

current_dir = os.getcwd()
dataset_path = current_dir + '/dataset/rtsd-frames/rtsd-frames/'
boxes_path = current_dir + '/dataset/'

train_image_path = current_dir + '/dataset/images/train'
train_label_path = current_dir + '/dataset/labels/train'

val_image_path = current_dir + '/dataset/images/val'
val_label_path = current_dir + '/dataset/labels/val'

label_dict = {'2_1': 0, '1_23': 1, '1_17': 2, '3_24_n40': 3, '8_2_1': 4, 
                '5_20': 5, '3_24_n20': 6, '5_19_1': 7, '5_16': 8, '3_25_n20': 9, 
                '6_16': 10, '7_15': 11, '2_2': 12, '2_4': 13, '8_13_1': 14, 
                '4_2_1': 15, '1_20_3': 16, '1_25': 17, '3_4_n8': 18, '8_3_2': 19, 
                '3_4_1': 20, '4_1_6': 21, '4_2_3': 22, '4_1_1': 23, '1_33': 24, 
                '5_15_5': 25, '3_27': 26, '1_15': 27, '4_1_2_1': 28, '6_3_1': 29, 
                '8_1_1': 30, '6_7': 31, '5_15_3': 32, '7_3': 33, '1_19': 34, '6_4': 35, 
                '8_1_4': 36, '8_8': 37, '1_16': 38, '1_11_1': 39, '6_6': 40, '5_15_1': 41, 
                '7_2': 42, '5_15_2': 43, '7_12': 44, '3_18': 45, '5_6': 46, '5_5': 47, 
                '7_4': 48, '4_1_2': 49, '8_2_2': 50, '7_11': 51, '3_24_n5': 52, '1_22': 53, 
                '1_27': 54, '2_3_2': 55, '5_15_2_2': 56, '1_8': 57, '3_13_r5': 58, '2_3': 59, 
                '8_3_3': 60, '2_3_3': 61, '7_7': 62, '1_11': 63, '8_13': 64, '3_24_n30': 65, 
                '1_12_2': 66, '1_20': 67, '1_12': 68, '3_24_n60': 69, '3_24_n70': 70, 
                '3_24_n50': 71, '3_32': 72, '2_5': 73, '3_1': 74, '4_8_2': 75, '3_20': 76, 
                '3_13_r4.5': 77, '3_2': 78, '2_3_6': 79, '5_22': 80, '5_18': 81, '2_3_5': 82, 
                '7_5': 83, '8_4_1': 84, '3_13_r3.7': 85, '3_14_r3.7': 86, '1_2': 87, '1_20_2': 88, 
                '4_1_4': 89, '7_6': 90, '8_1_3': 91, '8_3_1': 92, '4_3': 93, '4_1_5': 94, 
                '8_2_3': 95, '8_2_4': 96, '3_24_n80': 97, '1_31': 98, '3_10': 99, '4_2_2': 100, 
                '3_13_r2.5': 101, '7_1': 102, '3_28': 103, '4_1_3': 104, '5_4': 105, '5_3': 106, 
                '3_25_n40': 107, '3_13_r4': 108, '6_8_2': 109, '3_31': 110, '6_2_n50': 111, 
                '3_24_n10': 112, '3_25_n50': 113, '1_21': 114, '3_21': 115, '1_13': 116, '1_14': 117, 
                '6_2_n70': 118, '2_3_4': 119, '4_8_3': 120, '6_15_2': 121, '2_6': 122, '3_18_2': 123, 
                '4_1_2_2': 124, '1_7': 125, '3_19': 126, '1_18': 127, '2_7': 128, '8_5_4': 129, 
                '3_25_n80': 130, '5_15_7': 131, '5_14': 132, '5_21': 133, '1_1': 134, '6_15_1': 135, 
                '3_4_n2': 136, '8_6_4': 137, '8_15': 138, '4_5': 139, '3_13_r4.2': 140, '6_2_n60': 141,
                 '3_11_n23': 142, '3_11_n9': 143, '8_18': 144, '8_4_4': 145, '3_30': 146, '5_7_1': 147,
                  '5_7_2': 148, '1_5': 149, '3_29': 150, '6_15_3': 151, '5_12': 152, '3_16_n3': 153, 
                  '3_13_r4.3': 154, '1_30': 155, '5_11': 156, '1_6': 157, '8_6_2': 158, '6_8_3': 159, 
                  '3_12_n10': 160, '3_12_n6': 161, '3_33': 162, '3_11_n13': 163, '3_14_r2.7': 164, 
                  '3_16_n1': 165, '8_4_3': 166, '5_8': 167, '3_11_n20': 168, '3_11_n5': 169, '8_14': 170,
                   '3_11_n8': 171, '3_4_n5': 172, '8_17': 173, '3_6': 174, '3_14_r3': 175, '1_26': 176, 
                   '3_12_n5': 177, '8_5_2': 178, '6_8_1': 179, '5_17': 180, '1_10': 181, '3_13_r3.5': 182, 
                   '3_13_r3.3': 183, '3_13_r4.1': 184, '3_11_n17': 185, '8_16': 186, '3_13_r3': 187, 
                   '3_25_n70': 188, '6_2_n20': 189, '3_12_n3': 190, '3_14_r3.5': 191, '3_13_r3.9': 192,
                    '6_2_n40': 193, '3_13_r5.2': 194, '7_18': 195, '7_14': 196, '8_23': 197}

USE_IMAGE_COPY = True

for dir_name in [train_image_path, val_image_path, train_label_path, val_label_path]:
    os.makedirs(os.path.join(dir_name), exist_ok=True)

ready_train_files = []
ready_val_files = []

source_dir = os.path.join(dataset_path)
for i, row in df.iterrows():
    filename = row['filename']

    im = Image.open(os.path.join(dataset_path, filename))

    x_from = int(row['x_from'] + (int(row['width']) / 2)) / im.size[0]
    y_from = int(row['y_from'] + (int(row['height']) / 2)) / im.size[1] 
    width = int(row['width']) / im.size[0]
    height = int(row['height']) / im.size[1]
    sign_class = row['sign_class']
    sign_id = label_dict[sign_class]


    if filename in ready_train_files:
        dest_dir_label = os.path.join(train_label_path) 
        text = "\n{0} {1} {2} {3} {4}".format(sign_id, x_from, y_from, 
                                            width, height)

        my_file = open(os.path.join(dest_dir_label, filename[:-4] + ".txt"), "a")
        my_file.write(text)
        my_file.close()
    
    elif filename in ready_val_files:
        dest_dir_label = os.path.join(val_label_path)
        text = "\n{0} {1} {2} {3} {4}".format(sign_id, x_from, y_from, 
                                            width, height)

        my_file = open(os.path.join(dest_dir_label, filename[:-4] + ".txt"), "a")
        my_file.write(text)
        my_file.close()

    else:
        if i % 5 != 0:
            ready_train_files.append(filename)
            dest_dir_image = os.path.join(train_image_path)
            dest_dir_label = os.path.join(train_label_path) 
        else:
            ready_val_files.append(filename)
            dest_dir_image = os.path.join(val_image_path)
            dest_dir_label = os.path.join(val_label_path)

        if USE_IMAGE_COPY:
            im = im.resize((640, 640), Image.LANCZOS)
            im.save(os.path.join(dest_dir_image, filename))

            # shutil.copy(os.path.join(source_dir, filename), os.path.join(dest_dir_image, filename))

        text = "{0} {1} {2} {3} {4}".format(sign_id, x_from, y_from, 
                                            width, height)

        my_file = open(os.path.join(dest_dir_label, filename[:-4] + ".txt"), "w+")
        my_file.write(text)
        my_file.close()