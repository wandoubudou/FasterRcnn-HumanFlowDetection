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

# 修改：
# 日期  2019.4.28
# 修改者： 李博深
# 修改目标： 支持读取视频
# 视频---->已标注的图片

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
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer
from my_utils.StatisticsPeople import *
from multiprocessing.connection import Client
import time
import json

"""
    设置端口号，通过tcp协议建立连接
    地址需要去config中设置
"""
address = cfg.FLAGS2['address']
client = Client(address)
counter = 0
sum_time = 0

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

save_path = os.path.join(cfg.FLAGS2['data_dir'], 'save_videos')


def init_connection():
    client.send(cfg.FLAGS2['process_json_info_rate'])
    client.send(int(1/cfg.FLAGS2['process_speed'])-2)  #发送fps信息


def send_image(data):
    client.send(data)
    info = client.recv()
    print(info)

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    imgs_save_dir = os.path.join(cfg.FLAGS2['data_dir'], 'bbox_imgs_temp')

    inds = np.where(dets[:, -1] >= thresh)[0]
    print("总人数为: ",len(inds))
    cur_time = time.time()
    place = cfg.FLAGS2['cur_place']
    set_place(place)
    if statistic_num(cur_time,len(inds)) == False:
        print('该统计的时间或者人数为None，跳过本次统计....')
    if len(inds) == 0:  # 没有检测到人的情况，返回原来的图片，RGB通道要进行相应的调整
        return im
    # im = im[:, :, (2, 1, 0)]
    # fig, ax = plt.subplots(figsize=(12, 12))

    # ax.imshow(im, aspect='equal')
    for i in inds:  # inds是这张图中的总人数
        bbox = dets[i, :4]
        score = dets[i, -1]
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(60, 20, 220), thickness=2)  # 画矩形框人
        cv2.putText(im, '{:s} {:.3f}'.format(class_name, score), (int(bbox[0]), int(bbox[1] - 2)), 2, 1,
                    color=(255, 0, 0))
        # ax.add_patch(
        #     plt.Rectangle((bbox[0], bbox[1]),
        #                   bbox[2] - bbox[0],
        #                   bbox[3] - bbox[1], fill=False,
        #                   edgecolor='red', linewidth=3.5))
        # ax.text(bbox[0], bbox[1] - 2,               #在标签上添加信息  person + 概率
        #         '{:s} {:.3f}'.format(class_name, score),
        #         bbox=dict(facecolor='blue', alpha=0.5),
        #         fontsize=14, color='white')         #字的大小and颜色
    return im



def demo(sess, net, im):
    """Detect object classes in an image using pre-computed object proposals."""

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)  # boxes为走一遍网络后，读入图片，检测到所有可能的box
    timer.toc()
    sum_time = timer.total_time
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
    cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]  # cls_boxes为某个类型的所有box的集合
    cls_scores = scores[:, cls_ind]  # 四个坐标点
    dets = np.hstack((cls_boxes,  # 保存了所有的框的坐标and得分值
                      cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, NMS_THRESH)  # keep为nms非极大值抑制过滤后剩下的框
    dets = dets[keep, :]
    img = vis_detections(im, cls, dets, thresh=CONF_THRESH)
    return img,sum_time

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args


def split_video_to_image(video='', video_name='', jpg_name='', timeF=1):  # timeF=1  一帧一帧的截
    start_time = time.time()
    step = 0
    vc = cv2.VideoCapture(video)
    if vc.isOpened():
        print('video is opening...')
        rval, frame = vc.read()
    else:
        print('video {} load failed...'.format(video_name))
        return None  # 加载失败

    jpg_filename = jpg_name
    if not os.path.exists(jpg_filename):
        os.mkdir(jpg_filename)

    while rval:
        rval, frame = vc.read()
        step += 1
        if step % timeF == 0:
            im_name = os.path.join(jpg_filename, 'image{}.jpg'.format(step))
            cv2.imwrite(im_name, frame)

    vc.release()
    end_time = time.time()
    print('time cost is {}'.format(end_time - start_time))
    return True


if __name__ == '__main__':
    place = input('请输入当前位置信息:\n')
    cfg.FLAGS2['cur_place'] = place
    args = parse_args()

    demonet = args.demo_net
    dataset = args.dataset
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

    # 2019.4.28 修改存在缺陷，
    # 这个修改只是将视频截帧的部分直接拿过来，截到的图片也只是存在文件中
    # 下次修改计划，让截取到的图片直接送到网络中...
    cap = cv2.VideoCapture(0)
    init_connection()

    while 1:
        counter += 1
        if counter == cfg.FLAGS2['process_json_info_rate']:
            client.send(sum_time)
            print('fps = ',int(cfg.FLAGS2['process_json_info_rate']/sum_time))
            people_info = process_json_info(int(cfg.FLAGS2['process_json_info_rate']/sum_time))
            people_info = json.dumps(people_info,ensure_ascii=False)
            print(people_info)
            client.send(people_info)

            counter = 0
            sum_time=0


        ret, img = cap.read()
        im,t_sum_time = demo(sess, net, img)
        sum_time += t_sum_time
        # sum_time 做一个时间累计  用来计算并动态规定展示视频的帧率
        send_image(im)

        cv2.imshow("q for quit", im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    client.close()

