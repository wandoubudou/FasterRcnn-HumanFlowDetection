#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
#from nets.resnet_v1 import resnetv1
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer
import time

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

save_path = os.path.join(cfg.FLAGS2['data_dir'],'save_videos')


def vis_detections(im, class_name, dets, thresh=0.5,image_name=''):
    """Draw detected bounding boxes."""
    imgs_save_dir = os.path.join(cfg.FLAGS2['data_dir'], 'bbox_imgs_temp')

    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:          #没有检测到人的情况，返回原来的图片，RGB通道要进行相应的调整
        im_f = os.path.join(cfg.FLAGS2["data_dir"], 'demo/temp_jpg', image_name)
        im1 = cv2.imread(im_f)
        cv2.imwrite(os.path.join(imgs_save_dir, image_name),im1)
        return
    # im = im[:, :, (2, 1, 0)]
    # fig, ax = plt.subplots(figsize=(12, 12))

    # ax.imshow(im, aspect='equal')
    for i in inds:                  #inds是这张图中的总人数
        bbox = dets[i, :4]
        score = dets[i, -1]
        cv2.rectangle(im,(bbox[0], bbox[1]),(bbox[2],bbox[3]),color=(60,20,220),thickness=2)        #画矩形框人
        cv2.putText(im, '{:s} {:.3f}'.format(class_name, score),(int(bbox[0]), int(bbox[1]-2)), 2, 1, color=(255, 0, 0))
        # ax.add_patch(
        #     plt.Rectangle((bbox[0], bbox[1]),
        #                   bbox[2] - bbox[0],
        #                   bbox[3] - bbox[1], fill=False,
        #                   edgecolor='red', linewidth=3.5))
        # ax.text(bbox[0], bbox[1] - 2,               #在标签上添加信息  person + 概率
        #         '{:s} {:.3f}'.format(class_name, score),
        #         bbox=dict(facecolor='blue', alpha=0.5),
        #         fontsize=14, color='white')         #字的大小and颜色

    # ax.set_title(('{} detections with '
    #               'p({} | box) >= {:.1f}').format(class_name, class_name,thresh),fontsize=14)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.draw()


    if not os.path.exists(imgs_save_dir):
        os.mkdir(imgs_save_dir)
        print('create floder {} done...'.format(imgs_save_dir))

    # cv2.imshow('test',im)
    # cv2.waitKey(0)

    # plt.savefig(os.path.join(imgs_save_dir,image_name))
    cv2.imwrite(os.path.join(imgs_save_dir, image_name), im)
    # plt.show()
    # plt.close()


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.FLAGS2["data_dir"], 'demo/temp_jpg', image_name)
    im = cv2.imread(im_file)


    # Detect all object classes and regress object bounds
    timer = Timer()
    timer2 = Timer()
    timer2.tic()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)        #boxes为走一遍网络后，读入图片，检测到所有可能的box
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.1
    NMS_THRESH = 0.1
    # for cls_ind, cls in enumerate(CLASSES[1:]):
    #     cls_ind += 1  # because we skipped background
    #     cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
    #     cls_scores = scores[:, cls_ind]
    #     dets = np.hstack((cls_boxes,
    #                       cls_scores[:, np.newaxis])).astype(np.float32)
    #     keep = nms(dets, NMS_THRE    SH)
    #     dets = dets[keep, :]
    #     vis_detections(im, cls, dets, thresh=CONF_THRESH)

    cls_ind = 15
    cls = 'person'
    cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]     #cls_boxes为某个类型的所有box的集合
    cls_scores = scores[:, cls_ind]   #四个坐标点
    dets = np.hstack((cls_boxes,            #保存了所有的框的坐标and得分值
                      cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, NMS_THRESH)        #keep为nms非极大值抑制过滤后剩下的框
    dets = dets[keep, :]
    vis_detections(im, cls, dets, thresh=CONF_THRESH,image_name=image_name)
    timer2.toc()
    print('timer2 time cost is {}'.format(timer2.average_time))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args

def split_video_to_image(video = '',video_name='',jpg_name='',timeF=1):     #timeF=1  一帧一帧的截
    start_time = time.time()
    step = 0
    vc = cv2.VideoCapture(video)
    if vc.isOpened():
        print('video is opening...')
        rval, frame = vc.read()
    else:
        print('video {} load failed...'.format(video_name))
        return None     #加载失败

    jpg_filename = jpg_name
    if not os.path.exists(jpg_filename):
        os.mkdir(jpg_filename)

    while rval:
        rval, frame  = vc.read()
        step += 1
        if step%timeF == 0:
            im_name = os.path.join(jpg_filename, 'image{}.jpg'.format(step))
            cv2.imwrite(im_name,frame)

    vc.release()
    end_time = time.time()
    print('截取图片的 time cost is {}'.format(end_time-start_time))
    return True

if __name__ == '__main__':
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    #tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default', NETS[demonet][0])
    tfmodel = './default/voc_2007_trainval/default/vgg16_faster_rcnn_iter_20000.ckpt'
    """
    if not os.path.isfile(tfmodel):
        print(tfmodel)
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta')
    """

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    # elif demonet == 'res101':
        # net = resnetv1(batch_size=1, num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture(sess, "TEST", 21,
                            tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    timeF = cfg.FLAGS2['timeF']
    video_names = ['01.mp4','02.mp4']           #videos名字
    video_name = video_names[1]     #选择具体的一个视频
    video_path = os.path.join(cfg.FLAGS2['data_dir'],'test_videos',video_name)  #找到对应视频源文件
    img_path = os.path.join(cfg.FLAGS2['data_dir'], 'demo/temp_jpg')    #图片路径
    if split_video_to_image(video_path,video_name,jpg_name=img_path,timeF=timeF) :     #视频--->图片（存入demo/temp_jpg中）
        print('视频截帧完成....')
    #2019.4.28 修改存在缺陷，
    #这个修改只是将视频截帧的部分直接拿过来，截到的图片也只是存在文件中
    #下次修改计划，让截取到的图片直接送到网络中...
    
    im_names = os.listdir(img_path)     #修改，读取图片的方式，通过list出来的图片名称，排序后，读入图片，增加普适性
    for i in range(len(im_names)-1):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        im_name = 'image{}.jpg'.format((i+1)*timeF)
        print('Demo for data/demo/{}'.format(im_name))
        demo(sess, net, im_name)

