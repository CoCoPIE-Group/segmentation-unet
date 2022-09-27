import cv2
import numpy as np
import os


path = "/home/tmp00047/test_effnet/Unet-pytorch/results/unet/test/images/"

file_name_list = os.listdir(path)

label_img_name = []
th_img_name = []
crf_img_name = []

for e in file_name_list:
    if 'label.png' in e:
        label_img_name.append(e)
    elif 'th.png' in e:
        th_img_name.append(e)
    elif 'crf.png' in e:
        crf_img_name.append(e)

iou_score_list = []

for i in range(len(label_img_name)):
    print('computing {}th image..'.format(i+1))
    img1 = cv2.imread(path + label_img_name[i])
    img2 = cv2.imread(path + crf_img_name[i])

    intersection = np.logical_and(img1, img2)
    union = np.logical_or(img1, img2)
    iou_score = np.sum(intersection) / np.sum(union)

    iou_score_list.append(iou_score)


print(np.mean(iou_score_list))
