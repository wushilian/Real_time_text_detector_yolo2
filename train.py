import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from model import Yolo_v2
import ops
from Generater import Gen_char
import config as cfg
import utils
from sklearn.utils import shuffle
import random

if not os.path.exists(cfg.check_point):
    os.makedirs(cfg.check_point)



a,b,c=utils.read_data()

X,Y=ops.make_label_imgs(a,b,c)
yolo_net=Yolo_v2(train_phase=True)
sess=tf.Session()
saver=tf.train.Saver(max_to_keep=1)
ckpt=tf.train.latest_checkpoint(cfg.check_point)
sess.run(tf.global_variables_initializer())
saver.restore(sess,ckpt)

for i in range(1000000):
    index=random.sample(range(len(X)),cfg.batch_size)
    x,y=X[index],Y[index]
    feed={yolo_net.img:x/255.,yolo_net.label:y}

    sess.run(yolo_net.train_op,feed_dict=feed)

    if i%100==0:

        xy_loss,wh_loss,c_loss,loss=sess.run([yolo_net.loss_xy,yolo_net.loss_wh,yolo_net.loss_c,yolo_net.loss],feed_dict=feed)
        print('iteration:%d,xy_loss:%f,wh_loss:%f,c_loss:%f,loss:%f'%(i,xy_loss,wh_loss,c_loss,loss))
        saver.save(sess,cfg.check_point+'/yolo',global_step=i)
