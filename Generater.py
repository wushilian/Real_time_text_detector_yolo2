#coding=utf-8
import cv2
from PIL import Image,ImageDraw,ImageFont
import numpy as np
import os
import random
from config import classes
from matplotlib import  pyplot as plt
from ops import make_label_imgs

font_name=os.listdir('font')
fonts=[]
for name in font_name:#读取字体，大小为25到35
    size=30
    while size<=50:
        zz=ImageFont.truetype("./font/"+name, size, 0)
        fonts.append(zz)
        size+=1

def Overlap(region,target):#
    for i in range(len(region)):
        w=min(region[i][2],target[2])-max(region[i][0],target[0])
        h=min(region[i][3],target[3])-max(region[i][1],target[1])
        if not (w<0 or h<0):
            return True
    return False

def noise(image,mode=1):
    ratio=np.random.randint(30,80)*0.01#增加0.3到0.8的噪声


    noise_image = (image.astype(float)/255)+(np.random.random((400,400))*(np.random.random()*ratio))
    norm = (noise_image - noise_image.min())/(noise_image.max() - noise_image.min())
    if mode == 1:

        norm  =(norm * 255).astype(np.uint8)
    return norm

def augmentation(image):
    mode=np.random.randint(0,4)
    #mode=3
    if mode == 0:
        image = cv2.GaussianBlur(image, (5,5), np.random.randint(1, 10))

    if mode == 1:
        size = 5;
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        image = cv2.filter2D(image, -1, kernel_motion_blur)

        # kernal = []
        # image =
    if mode == 2:
        size = 5;
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[:, int((size - 1) / 2)] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        image = cv2.filter2D(image, -1, kernel_motion_blur)
    if mode==8:
        image=noise(image)
    if mode==4:
        num_point=np.random.randint(10,100)
        for i in range(num_point):
            x_point=np.random.randint(0,400)
            y_point=np.random.randint(0,400)
            image[x_point,y_point]=np.random.randint(0,10)

    return image

def Draw_char():
    image = np.zeros(shape=(400, 400), dtype=np.uint8) + np.random.randint(100, 170)
    im = Image.fromarray(image)
    draw=ImageDraw.Draw(im)
    max_lines=7#max lines in image
    min_lines=5
    curr_lines=random.randint(min_lines,max_lines+1)
    curr_fonts=[]   #all lines fonts
    cur_height=[0]
    max_height=0    #the sum of fonts size
    rois=[]
    category=[]
    dict=''
    for i in range(curr_lines):
        font = random.choice(fonts)
        curr_fonts.append(font)
        max_height+=font.getsize('H')[1]+random.randint(1,5)
        cur_height.append(max_height)

    for i in range(curr_lines):
        x_point = random.randint(2, 10)
        font=curr_fonts[i]
        txt=random.sample(classes,random.randint(10,15))
        for j in range(len(txt)):
            txt_img=Image.new('L',(font.size+2,font.size+2),0)
            d=ImageDraw.Draw(txt_img)
            d.text((0,0),txt[j],font=font,fill=random.randint(0,30))
            angle=random.randint(-5,5)
            wa=txt_img.rotate(angle=angle,expand=1)
            mask_img = Image.new('L', (font.size+2, font.size+2), 0)
            mask = ImageDraw.Draw(mask_img)
            mask.text((0, 0), txt[j], font=font, fill=255)
            mask = mask_img.rotate(angle=angle,expand=1)

            w,h=font.getsize(txt[j])
            x_, y_ = font.getoffset(txt[j])

            im.paste(wa,(x_point,cur_height[i]),mask)
            #draw.rectangle(((x_point + x_, cur_height[i] + y_), (x_point + w, cur_height[i] + h)))
            rois.append([x_point + x_+int(w/2), cur_height[i] + y_+int(h/2),w,h])
            #category.append(txt[j])
            category.append(classes.index(txt[j]))
            #category.append(0)
            x_point+=w+x_#+random.randint(0,5)
    im=np.array(im)
    p = augmentation(im)
    ratio = np.random.randint(1, 20) * 0.01  # 增加0.3到0.8的噪声
    p = (p.astype(float) / 255) * (1 - ratio) + (np.random.random((400, 400)) * (np.random.random() * ratio))
    p = p * 255

    '''cv2.imwrite('img/' + image_name + '.jpg', p)
    f = open('labels/' + image_name + '.txt', 'w')
    f.write(dict)
    f.close()'''
    #plt.imshow(im,cmap='gray')
    #plt.show()
    return p,rois,category

def Gen_char(batch_size=32):
    while True:
        imgs=[]
        rois=[]
        labels=[]
        for i in range(batch_size):
            img,roi,cate=Draw_char()
            imgs.append(img);rois.append(roi);labels.append(cate)
        x,y=make_label_imgs(np.array(imgs),rois,labels)
        yield x[...,np.newaxis],y


if __name__=='__main__':
    gen=Gen_char()
    x,y=next(gen)
    cv2.imwrite('1.jpg',x[0])
