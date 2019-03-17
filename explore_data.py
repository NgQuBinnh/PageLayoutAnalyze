import sys
from read_anntations import *

import os
import cv2
from os.path import join as osj
import shutil
import json

fileList = getFileList('D:\Data\POD\Annotations')
root_path = 'D:\Coding\TableTrainNet\dataset'

data_set = open('D:\Coding\\tensorflow-deeplab-resnet\dataset\\test.txt','r').readlines()
images = [image.split(' ')[0].replace("/JpegImages/", "").replace(".jpg","") for image in data_set]

save_dir = 'D:\Coding\Object-Detection-Metrics\groundtruths'

for idx, image in enumerate(images):
    if idx == 38:
        break
    xml = 'D:\Data\POD\Annotations\\' + image + '.xml'
    ws = open(image + ".txt","w")
    data = FileData()
    data.readtxt(xml)
    bbx = []
    for line in data.pageLines:
        rect = line.rect
        bbx.append([rect.l, rect.u, rect.r, rect.d])
        print(line.kind)
        ws.write(line.kind + " " + str(rect.l) + " " + str(rect.u) + " " + str(rect.r) + " " + str(rect.d) +"\n")
    ws.close()

max_x = 0
max_y = 0

min_formula = 1000000000
min_figure = 1000000000
min_table = 1000000000

for txtname in fileList:
    txtpath = osj('D:\Data\POD\Annotations', txtname)

    data = FileData()
    data.readtxt(txtpath)
    for bb in data.pageLines:
        cate = bb.kind
        if getColor(kinds.index(cate)) == 1 and bb.rect.area() > 590:
            min_formula = min(min_formula, bb.rect.area())
        else:
            if getColor(kinds.index(cate)) == 2 and bb.rect.area() > 1500:
                min_figure = min(min_figure, bb.rect.area())
            else:
                if getColor(kinds.index(cate)) == 3:
                    min_table = min(min_table, bb.rect.area())


print("Min formula = " + str(min_formula))
print("Min figure = " + str(min_figure))
print("Min table = " + str(min_table))
