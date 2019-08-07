import cv2
import glob
import os
import matplotlib.pyplot as plt
import lib.config.config as cfg
import numpy as np

# fps = 30
size = (1920,1080)
timeF = 1
save_path = os.path.join(cfg.FLAGS2['data_dir'],'save_videos/')
img_path = os.path.join(cfg.FLAGS2['data_dir'],'bbox_imgs_temp')

def con_im_to_video(timeF,fps,size,name):

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    print('img_path = ', img_path)

    video_writer = cv2.VideoWriter(save_path+name,cv2.VideoWriter_fourcc('M','P','4','2'),fps,size)
    for i in range(len(os.listdir(img_path))):
        im = os.path.join(img_path,'image{}.jpg'.format((i+1)*timeF))
        print(im)
        # read_img = cv2.imdecode(np.fromfile(im,dtype=np.uint8),cv2.IMREAD_COLOR)
        read_img = cv2.imread(im)

        video_writer.write(read_img)
    video_writer.release()

    path = img_path
    names = os.listdir(path)
    for name in names:
        os.remove(os.path.join(img_path,name))      #清空使用过的图片
    return True

# im = cv2.imread('E:\FasterRcnn\Faster-RCNN-TensorFlow-Python3.5-master\data/bbox_imgs_temp\image44.jpg')
#
# cv2.rectangle(im,(100,100),(200,200),color=(60,20,220),thickness=2)
# cv2.putText(im, '{:s} {:.3f}'.format('aa', 0.9),(100,90),2,0.5,color=(255, 0, 0))
#
#
# cv2.imshow('test',im)
# cv2.waitKey(0)




