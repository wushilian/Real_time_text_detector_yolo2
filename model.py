import tensorflow as tf
import config as cfg
from ops import slice_tensor,smooth_L1

class Yolo_v2():
    def __init__(self,img=None,label=None,train_phase=True):
        if img is None:
            self.img=tf.placeholder(shape = [None, cfg.img_height,cfg.img_width,cfg.img_channel], dtype=tf.float32, name='image_placeholder')
        else:
            self.img=img

        if label is None:
            self.label = tf.placeholder(shape=[None,cfg.Grid_h, cfg.Grid_w, cfg.num_anchors, 6], dtype=tf.float32,name='label_palceholder')
        else:
            self.label=label

        with tf.name_scope('backbone'):
            features,self.pred=self.yolo_net(self.img,train_phase=train_phase)
        with tf.name_scope('loss'):
            self.loss=self.yolo_loss(pred=self.pred,label=self.label,lambda_coord=cfg.lambda_coor,lambda_no_obj=cfg.lambda_obj)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op=tf.train.AdamOptimizer(learning_rate=cfg.lr).minimize(self.loss)






    def yolo_net(self,x,train_phase=True):
        features=[]

        net = tf.layers.conv2d(x, 32, [3, 3], padding='SAME', activation=tf.nn.relu, use_bias=False,kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        net=tf.layers.batch_normalization(net,training=train_phase)
        net=tf.nn.relu(net)
        features.append(net)

        net = tf.layers.max_pooling2d(net, [2, 2], [2, 2])
        net = tf.layers.conv2d(net, 64, [3, 3], padding='SAME', use_bias=False, activation=tf.nn.relu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        net = tf.layers.batch_normalization(net, training=train_phase)
        net = tf.nn.relu(net)
        features.append(net)

        net = tf.layers.max_pooling2d(net, [2, 2], [2, 2])
        net = tf.layers.conv2d(net, 128, [3, 3], padding='SAME', use_bias=False, activation=tf.nn.relu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        net = tf.layers.batch_normalization(net, training=train_phase)
        net = tf.nn.relu(net)
        features.append(net)

        net = tf.layers.max_pooling2d(net, [2, 2], [2, 2])
        net = tf.layers.conv2d(net, 256, [3, 3], padding='SAME', use_bias=False, activation=tf.nn.relu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        net = tf.layers.batch_normalization(net, training=train_phase)
        net = tf.nn.relu(net)
        features.append(net)

        net = tf.layers.max_pooling2d(net, [2, 2], [2, 2])
        net = tf.layers.conv2d(net, 512, [3, 3], padding='SAME', use_bias=False, activation=tf.nn.relu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        net = tf.layers.batch_normalization(net, training=train_phase)
        net = tf.nn.relu(net)
        features.append(net)

        net=tf.layers.conv2d(net,filters=cfg.num_anchors*(cfg.num_classes+5),kernel_size=[1,1])

        cnn_shape=net.get_shape().as_list()
        print('Grid_h:%d,Grid_w:%d'%(cnn_shape[1],cnn_shape[2]))
        y = tf.reshape(net, shape=(-1,cfg.Grid_h, cfg.Grid_w, cfg.num_anchors,cfg.num_classes+5), name='y')
        return features,y


    def yolo_loss(self,pred, label, lambda_coord, lambda_no_obj):
        mask = slice_tensor(label, 5)
        label = slice_tensor(label, 0, 4)

        mask = tf.cast(tf.reshape(mask, shape=(-1,cfg.Grid_h, cfg.Grid_w, cfg.num_anchors)), tf.bool)

        with tf.name_scope('mask'):
            masked_label = tf.boolean_mask(label, mask)
            masked_pred = tf.boolean_mask(pred, mask)
            neg_masked_pred = tf.boolean_mask(pred, tf.logical_not(mask))

        with tf.name_scope('pred'):
            masked_pred_xy = tf.sigmoid(slice_tensor(masked_pred, 0, 1))
            masked_pred_wh = tf.exp(slice_tensor(masked_pred, 2, 3))
            masked_pred_o = tf.sigmoid(slice_tensor(masked_pred, 4))
            masked_pred_no_o = tf.sigmoid(slice_tensor(neg_masked_pred, 4))
            masked_pred_c = tf.nn.softmax(slice_tensor(masked_pred, 5, 41))


        with tf.name_scope('lab'):
            masked_label_xy = slice_tensor(masked_label, 0, 1)
            masked_label_wh = slice_tensor(masked_label, 2, 3)
            masked_label_c = slice_tensor(masked_label, 4)
            masked_label_c_vec = tf.reshape(tf.one_hot(tf.cast(masked_label_c, tf.int32), depth=cfg.num_classes),
                                            shape=(-1, cfg.num_classes))

        with tf.name_scope('merge'):
            with tf.name_scope('loss_xy'):
                #loss_xy = tf.reduce_mean(tf.square(masked_pred_xy - masked_label_xy))
                self.loss_xy = tf.reduce_mean(smooth_L1(masked_pred_xy - masked_label_xy))
            with tf.name_scope('loss_wh'):
                #loss_wh = tf.reduce_mean(tf.square(masked_pred_wh - masked_label_wh))
                self.loss_wh=tf.reduce_mean(smooth_L1(masked_pred_wh - masked_label_wh))
            with tf.name_scope('loss_obj'):
                loss_obj = tf.reduce_mean(tf.square(masked_pred_o - 1))
            with tf.name_scope('loss_no_obj'):
                loss_no_obj = tf.reduce_mean(tf.square(masked_pred_no_o))
            with tf.name_scope('loss_class'):
                loss_c=tf.nn.softmax_cross_entropy_with_logits(labels=masked_label_c_vec,logits=masked_pred_c)
                self.loss_c=tf.reduce_mean(loss_c)
                #loss_c = tf.reduce_mean(tf.square(masked_pred_c - masked_label_c_vec))

            loss = lambda_coord * (self.loss_xy + self.loss_wh) + loss_obj + lambda_no_obj * loss_no_obj + self.loss_c

        return loss

if __name__=='__main__':
    yolo=Yolo_v2()
