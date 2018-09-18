import config as cfg
import cv2
import os
import numpy as np

def read_data(data_dir='data'):
    images = []
    coors=[]
    labels=[]
    files=os.listdir(os.path.join(data_dir,'label'))
    for file_id in files:
        image_name = file_id.split('.tx')[0] + ".bmp"
        label_name = file_id
        image_path = "data/img/" + image_name
        label_path = "data/label/" + label_name
        img = cv2.imread(image_path,0)
        img=cv2.resize(img,(cfg.img_width,cfg.img_height))
        images.append(img[...,np.newaxis])
        with open(label_path, "r") as label_r:
            coor = []
            label=[]
            for line in label_r.readlines():
                linestrlist = line.strip().split(" ")

                linelist = [float(i) for i in linestrlist]
                linelist[1] = linelist[1] * 400
                linelist[3] = linelist[3] * 400
                linelist[2] = linelist[2] * 400
                linelist[4] = linelist[4] * 400
                linelist[1] = linelist[1] - linelist[3] / 2
                linelist[2] = linelist[2] - linelist[4] / 2
                label.append(int(linelist[0]))
                coor.append(linelist[1:5])

        coors.append(coor)
        labels.append(label)
    return images,coors,labels


