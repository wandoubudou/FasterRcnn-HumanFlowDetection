# FasterRcnn-HumanFlowDetection


## my python version is 3.5.4
### 这是一个2018年软件杯赛题，我完成了他的要求并且将其公开。


### This is a project for learning faster rcnn ,just for fun, if you want to use my module, you can download it from BaiDuYun Disk  here is the link:https://pan.baidu.com/s/1zgfpZajZpfPsswSrLoXzCQ
### 数据集和模型太大了，我把它挂到百度网盘上了，如果想用我的数据集和模型，可以去下载   https://pan.baidu.com/s/1zgfpZajZpfPsswSrLoXzCQ


### the ability this program can do is to detect a video and mark all the person inside and save the person number info in data folder.
### 这个程序可以视频--->视频   需要检测的视频经过 video_dect.py 渲染可以识别出视频内所有人得信息，以json格式保存在data的save_videos文件夹中
### json文件保存的是人数信息，每隔一秒统计一次，以一秒内平均人员信息作为最终结果
### 你只需要把需要检测的视频放在 data\test_videos下面，然后去lib\config.py中修改FLAGS2['video_name']视频名称即可运行



### 此外 这个程序实现了两台电脑的传输功能，也就是说，可以远程检测，
### 比如电脑A运行people_detect.py，电脑B就会收到电脑A摄像头捕获到的实时检测后的信息，传输协议是TCP
### 再运行之前，需要配置对应电脑的ip和端口，在lib\config.py的FLAGS2['address']配置。


### 当然你也可以修改我一小部分的代码，关闭远程连接，自己随便玩玩


the final project structure may be like this:

![alt text](https://github.com/wandoubudou/FasterRcnn-HumanFlowDetection/blob/master/images/project.png "project")






## here are some result demo
![alt text](https://github.com/wandoubudou/FasterRcnn-HumanFlowDetection/blob/master/images/1.png "project")




![alt text](https://github.com/wandoubudou/FasterRcnn-HumanFlowDetection/blob/master/images/2.png "project")



### if you get bugs ,please fork me
### 有bug请找我