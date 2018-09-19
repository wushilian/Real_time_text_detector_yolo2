import tensorflow as tf
import config as cfg
import numpy as np
import cv2


def smooth_L1(x):
    loss = tf.where(tf.less(tf.abs(x),1), tf.square(x) * 0.5, tf.abs(x) - 0.5)
    return loss

def slice_tensor(x, start, end=None):
    if end is None:
        end=start
        y = x[..., start:end + 1]
    else:
        y = x[..., start:end + 1]

    return y


def iou_wh(r1, r2):
    '''
    cal iou
    :param r1: [w,h]
    :param r2: [w,h]
    :return: float
    '''
    min_w = min(r1[0], r2[0])
    min_h = min(r1[1], r2[1])
    area_r1 = r1[0] * r1[1]
    area_r2 = r2[0] * r2[1]

    intersect = min_w * min_h
    union = area_r1 + area_r2 - intersect

    return intersect / union


def get_grid_cell(roi):
    x_center = roi[0] + roi[2]/ 2.0
    y_center = roi[1] + roi[3]/ 2.0

    grid_x = int(x_center / float(cfg.img_width) * float(cfg.Grid_w))
    grid_y = int(y_center / float(cfg.img_height) * float(cfg.Grid_h))

    return grid_x, grid_y


def get_active_anchors(roi, anchors,iou_th=0.7):
    '''
    calculate the iou of roi whith each anchors,and return the  match index of anchors
    :param roi: [x,y,w,h]
    :param anchors: [[w,h],[w,h]...]
    :param iou_th: the threshold
    :return: [index,index...]
    '''
    indxs = []
    iou_max, index_max = 0, 0
    for i, a in enumerate(anchors):
        iou = iou_wh(roi[2:], a)
        if iou > iou_th:
            indxs.append(i)
        if iou > iou_max:
            iou_max, index_max = iou, i

    if len(indxs) == 0:
        indxs.append(index_max)

    return indxs





def roi2label(roi, anchor, raw_w, raw_h, grid_w, grid_h):
    '''

    :param roi: [x1,y1,x2,y2]
    :param anchor: [w,h]
    :param raw_w: weight of the image
    :param raw_h: height of the image
    :param grid_w:
    :param grid_h:
    :return:
    '''
    x_center = roi[0] + roi[2]/ 2.0
    y_center = roi[1] + roi[3]/ 2.0

    grid_x = x_center / float(raw_w) * float(grid_w)
    grid_y = y_center / float(raw_h) * float(grid_h)

    grid_x_offset = grid_x - int(grid_x)
    grid_y_offset = grid_y - int(grid_y)

    roi_w_scale = roi[2] / anchor[0]
    roi_h_scale = roi[3] / anchor[1]

    label = [grid_x_offset, grid_y_offset, roi_w_scale, roi_h_scale]

    return label


def onehot(idx, num):
    ret = np.zeros([num], dtype=np.float32)
    ret[idx] = 1.0

    return ret


def make_label_imgs(imgs, rois, classes):
    anchors = cfg.anchors
    n_anchors = np.shape(anchors)[0]

    labels=[]
    for img, rois, classes in zip(imgs, rois, classes):

        rois = np.array(rois, dtype=np.float32)
        classes = np.array(classes, dtype=np.int32)


        raw_h =cfg.img_height
        raw_w = cfg.img_width
        #img = cv2.resize(img, (cfg.img_width,cfg.img_height))

        label = np.zeros([cfg.Grid_h,cfg.Grid_w, n_anchors, 6], dtype=np.float32)

        for roi, cls in zip(rois, classes):
            roi[2]=roi[2]*cfg.Grid_w/cfg.img_width
            roi[3]=roi[3]*cfg.Grid_h/cfg.img_height

            active_indxs = get_active_anchors(roi, anchors)
            #print(active_indxs)
            grid_x, grid_y = get_grid_cell(roi)
            if grid_x<cfg.Grid_w and grid_y<cfg.Grid_h:
                #print(grid_x,grid_y,roi)
                for active_indx in active_indxs:
                    anchor_label = roi2label(roi, anchors[active_indx], raw_w, raw_h,cfg.Grid_w,cfg.Grid_h)
                    label[grid_y, grid_x, active_indx] = np.concatenate((anchor_label, [cls], [1.0]))

        labels.append(label)
    return np.array(imgs),np.array(labels)
