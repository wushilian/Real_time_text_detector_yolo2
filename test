import tensorflow as tf
import numpy as np
import cv2
import config as cfg
from model import Yolo_v2
from matplotlib import pyplot as plt





def iou(r1, r2):
    intersect_w = np.maximum(np.minimum(r1[0] + r1[2], r2[0] + r2[2]) - np.maximum(r1[0], r2[0]), 0)
    intersect_h = np.maximum(np.minimum(r1[1] + r1[3], r2[1] + r2[3]) - np.maximum(r1[1], r2[1]), 0)
    area_r1 = r1[2] * r1[3]
    area_r2 = r2[2] * r2[3]
    intersect = intersect_w * intersect_h
    union = area_r1 + area_r2 - intersect

    return intersect / union


def softmax(x):
    e_x = np.exp(x)
    return e_x / np.sum(e_x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def preprocess_data(data, anchors):
    locations = []
    classes = []
    for i in range(cfg.Grid_h):
        for j in range(cfg.Grid_w):
            for k in range(cfg.num_anchors):
                class_vec = softmax(data[0, i, j, k, 5:])
                objectness = sigmoid(data[0, i, j, k, 4])
                class_prob = objectness * class_vec

                scale_w =cfg.img_width*1.0/cfg.Grid_w
                scale_h = cfg.img_height*1.0/cfg.Grid_h

                w = np.exp(data[0, i, j, k, 2]) * anchors[k][0]# * scale_w
                h = np.exp(data[0, i, j, k, 3]) * anchors[k][1]# * scale_h
                dx = sigmoid(data[0, i, j, k, 0])
                dy = sigmoid(data[0, i, j, k, 1])
                x = (j + dx) * scale_w - w / 2.0
                y = (i + dy) * scale_h - h / 2.0

                classes.append(class_prob)
                locations.append([x, y, w, h])

    classes = np.array(classes)
    locations = np.array(locations)

    return classes, locations


def non_max_supression(classes, locations):
    classes = np.transpose(classes)
    indxs = np.argsort(-classes, axis=1)

    for i in range(classes.shape[0]):
        classes[i] = classes[i][indxs[i]]

    for class_idx, class_vec in enumerate(classes):
        for roi_idx, roi_prob in enumerate(class_vec):

            if roi_prob < cfg.prob_th:
                classes[class_idx][roi_idx] = 0

    for class_idx, class_vec in enumerate(classes):
        for roi_idx, roi_prob in enumerate(class_vec):

            if roi_prob == 0:
                continue
            roi = locations[indxs[class_idx][roi_idx]]

            for roi_ref_idx, roi_ref_prob in enumerate(class_vec):

                if roi_ref_prob == 0 or roi_ref_idx <= roi_idx:
                    continue

                roi_ref = locations[indxs[class_idx][roi_ref_idx]]

                if iou(roi, roi_ref) > cfg.iou_th:
                    classes[class_idx][roi_ref_idx] = 0

    return classes, indxs


def draw(classes, rois, indxs, img):



    for class_idx, c in enumerate(classes):
        for loc_idx, class_prob in enumerate(c):

            if class_prob > 0:

                x = int(rois[indxs[class_idx][loc_idx]][0] )
                y = int(rois[indxs[class_idx][loc_idx]][1])
                w = int(rois[indxs[class_idx][loc_idx]][2])
                h = int(rois[indxs[class_idx][loc_idx]][3])
                #print(x,y,w,h)
                img=cv2.rectangle(img, (x, y), (x + w, y + h), (255,255,0), 4)
                #font = cv2.FONT_HERSHEY_SIMPLEX
                #text = names[class_idx] + ' %.2f' % class_prob
                #cv2.putText(img, text, (x, y - 8), font, 0.7, colors[class_idx], 2, cv2.LINE_AA)

    return img
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def test():
    yolo=Yolo_v2(train_phase=False)
    anchors = cfg.anchors

    sess = tf.Session()
    saver = tf.train.Saver()
    ckpt=tf.train.latest_checkpoint(cfg.check_point)
    saver.restore(sess, ckpt)

    img=cv2.imread('1.jpg',0)
    img_for_net = cv2.resize(img, (cfg.img_width,cfg.img_height))
    img_for_net = img_for_net / 255.0
    data = sess.run(yolo.pred, feed_dict={yolo.img:np.array([img_for_net[...,np.newaxis]])})

    classes, rois = preprocess_data(data, anchors)
    #print(classes.shape)
    classes, indxs = non_max_supression(classes, rois)
    img=draw(classes, rois, indxs, img)

    plt.imshow(img)
    plt.show()



if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    test()
