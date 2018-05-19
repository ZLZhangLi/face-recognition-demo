# -*- coding: utf-8 -*-
import numpy as np
import os
import cv2
import cv2.cv as cv
from skimage import transform as tf
from PIL import Image, ImageDraw
import threading
from time import ctime,sleep
import time
import sklearn
import matplotlib.pyplot as plt
import skimage
import sys
import sklearn.metrics.pairwise as pw
from PyQt5 import QtWidgets,QtGui
from PyQt5.QtWidgets import (QWidget, QPushButton, QLineEdit,
    QInputDialog, QApplication, QMainWindow)
from PyQt5.QtGui import *
from PyQt5.QtCore import *

#caffe_root = '/home/zhangli/caffe/'
caffe_root = 'F:/caffe/caffe-master/Build/x64/Release/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

#保存人脸的位置
global face_rect
face_rect=[]
global pattern
global reg_id
global ID
pattern = 0
ID = 0
reg_id = 0
#caffe.set_device(0)
global net
#net=caffe.Classifier('F:/vgg_face_caffe/vgg_face_caffe/VGG_FACE_deploy.prototxt',
#    'F:/vgg_face_caffe/vgg_face_caffe/VGG_FACE.caffemodel')
#net=caffe.Classifier('/media/zhangli/program/vgg_face_caffe/vgg_face_caffe/VGG_FACE_deploy.prototxt',
#    '/media/zhangli/program/vgg_face_caffe/vgg_face_caffe/VGG_FACE.caffemodel')
net=caffe.Classifier('./model/ResNet-50-deploy.prototxt','./model/ResNet-50-model.caffemodel')
def detect(img, cascade):
    # CV_HAAR_SCALE_IMAGE，按比例正常检测
    rects = cv.HaarDetectObjects(img, cascade, cv.CreateMemStorage(), 1.1, 2, cv.CV_HAAR_DO_CANNY_PRUNING,
                                 (255, 255))
    if len(rects) == 0:
        return []
    result = []
    # 将检测到的位置保存到result
    for r in rects:
        result.append((r[0][0], r[0][1], r[0][0] + r[0][2], r[0][1] + r[0][3]))
    # 返回人脸的位置和大小,大小限定在300~500之间
    if result[0][2] > 300 and result[0][3] > 300 and result[0][2] < 500 and result[0][3] < 500:
        return result
    else:
        return []
# 画绿色的人脸框
def draw_rects(img, rects, color):
    if rects:
        for i in rects:
            # 画一个绿色的矩形框
            cv.Rectangle(img, (int(rects[0][0]), int(rects[0][1])), (int(rects[0][2]), int(rects[0][3])),
                         cv.CV_RGB(0, 255, 0), 1, 8, 0)
# 用来注册一个用户
def register(path, img, rects):
    if rects:
        # 保证图片是N*N的,即正方形
        if rects[0][2] < rects[0][3]:
            cv.SetImageROI(img, (rects[0][0] + 10, rects[0][1] + 10, rects[0][2] - 50, rects[0][2] - 50))
        else:
            cv.SetImageROI(img, (rects[0][0] + 10, rects[0][1] + 10, rects[0][3] - 50, rects[0][3] - 50))
        dst = cv.CreateImage((224, 224), 8, 3)
        # 保存人脸
        cv.Resize(img, dst, cv.CV_INTER_LINEAR)
        cv.SaveImage(path, dst)
# 用来识别一个用户
def recog(md, img):
    global face_rect
    src_path = './regist_pic/' + str(md)
    while True:
        rects = face_rect
        if rects:
            # img保存用来验证的人脸
            if rects[0][2] < rects[0][3]:
                cv.SetImageROI(img, (rects[0][0] + 10, rects[0][1] + 10, rects[0][2] - 100, rects[0][2] - 100))
            else:
                cv.SetImageROI(img, (rects[0][0] + 10, rects[0][1] + 10, rects[0][3] - 100, rects[0][3] - 100))
            # 将img暂时保存起来
            dst = cv.CreateImage((224, 224), 8, 3)
            cv.Resize(img, dst, cv.CV_INTER_LINEAR)
            cv.SaveImage('./temp.bmp', dst)
            # 取出5张注册的人脸,分别与带验证的人脸进行匹配,可以得到五个相似度,保存到scores中
            scores = []
            for i in range(2):
                res = compar_pic('./temp.bmp', src_path + '/' + str(i) + '.bmp')
                scores.append(res)
                print res
            # 求scores的均值
            result = avg(scores)
            print 'avg is :', avg(scores)
            return result
def avg(scores):
    max = scores[0]
    min = scores[0]
    res = 0.0
    for i in scores:
        res = res + i
        if min > i:
            min = i
        if max < i:
            max = i
    return (max + min) / 2

def compar_pic(path1, path2):
    global net
    # 加载验证图片
    X = read_image(path1)
    test_num = np.shape(X)[0]
    # X  作为 模型的输入
    out = net.forward_all(data=X)
    # fc7是模型的输出,也就是特征值
    # feature1 = np.float64(out['fc7'])
    feature1 = np.float64(net.blobs['fc1000'].data)
    feature1 = np.reshape(feature1, (test_num, 1000))
    # np.savetxt('feature1.txt', feature1, delimiter=',')
    # 加载注册图片
    X = read_image(path2)
    # X  作为 模型的输入
    out = net.forward_all(data=X)
    # fc7是模型的输出,也就是特征值
    # feature2 = np.float64(out['fc7'])
    feature2 = np.float64(net.blobs['fc1000'].data)
    feature2 = np.reshape(feature2, (test_num, 1000))
    # np.savetxt('feature2.txt', feature2, delimiter=',')
    # 求两个特征向量的cos值,并作为是否相似的依据
    predicts = pw.cosine_similarity(feature1, feature2)
    return predicts

def read_image(filelist):
    averageImg = [129.1863, 104.7624, 93.5940]
    X = np.empty((1, 3, 224, 224))
    word = filelist.split('\n')
    filename = word[0]
    im1 = skimage.io.imread(filename, as_grey=False)
    image = skimage.transform.resize(im1, (224, 224)) * 255
    X[0, 0, :, :] = image[:, :, 0] - averageImg[0]
    X[0, 1, :, :] = image[:, :, 1] - averageImg[1]
    X[0, 2, :, :] = image[:, :, 2] - averageImg[2]
    return X
# 用来显示当前图片
# Opencv中人脸检测的一个级联分类器
cascade = cv.Load("./haarcascade_frontalface_alt.xml")
# 获取视频流的接口，0表示摄像头的id号，当只连接一个摄像头时默认为0
cam = cv.CaptureFromCAM(0)
'''def show_img():
    global face_rect
    # 一个死循环，用来不间断的显示图片
    while True:
        img = cv.QueryFrame(cam)  # 取出视频中的一帧
        # 保存三通道的图片
        src = cv.CreateImage((img.width, img.height), 8, 3)
        cv.Resize(img, src, cv.CV_INTER_LINEAR)
        # 保存灰度图片
        gray = cv.CreateImage((img.width, img.height), 8, 1)
        cv.CvtColor(img, gray, cv.CV_BGR2GRAY)  # 将rgb图片变成灰度图
        cv.EqualizeHist(gray, gray)  # 对灰度图进行直方图均衡化
        rects = detect(gray, cascade)  # 传入图片和分类器，如果检测到人脸，返回人脸的坐标和大小
        face_rect = rects
        # 话那个绿色的人脸框
        draw_rects(src, rects, (0, 255, 0))
        # 显示画框的人脸
        cv.ShowImage('DeepFace ZhangLi', src)
        cv2.waitKey(5) == 27
    cv2.destroyAllWindows()

t1 = threading.Thread(target=show_img)'''
# Iterate file system.
def loadInfo(dataset_root):
    folders = sorted(os.listdir(dataset_root))
    file_info = []
    for folder in folders:
        file_info.append(folder)
    return file_info
class Example(QMainWindow):
    def __init__(self):
        super(Example, self).__init__()
        #QtGui.QWidget.__init__(self, Example)
        self.initUI()
    def initUI(self):
        self.img1 = QtWidgets.QLabel(self)
        self.img1.resize(224,224)
        self.img1.move(715,20)

        self.img2 = QtWidgets.QLabel(self)
        self.img2.resize(224,224)
        self.img2.move(960,20)

        self.show_video = QtWidgets.QLabel(self)
        self.show_video.resize(750,750)
        self.show_video.move(20,20)

        self.label1 = QtWidgets.QLabel(self)
        self.label1.resize(150,25)
        self.label1.setFont(QFont("华文新魏", 20))
        self.label1.move(1200,40)

        self.label2 = QtWidgets.QLabel(self)
        self.label2.resize(150,25)
        self.label2.setFont(QFont("华文新魏", 20))
        self.label2.move(1200,80)
        #self.label_XX.setStyleSheet("color:rgb(255,235,205)")

        self.label3 = QtWidgets.QLabel(self)
        self.label3.resize(150,25)
        self.label3.setFont(QFont("华文新魏", 20))
        self.label3.move(1200,120)
        #self.label_XX.setStyleSheet("color:rgb(255,235,205)")

        self.label_XX = QtWidgets.QLabel(self)
        self.label_XX.resize(150,25)
        self.label_XX.setFont(QFont("华文新魏", 20))
        self.label_XX.move(1200,10)
        self.label_XX.setText('个人信息')
        #self.label_XX.setStyleSheet(QtWidgets.QLabel{"color:rgb(255,235,205)"})

        '''self.label_REC = QtWidgets.QLabel(self)
        self.label_REC.resize(200,40)
        self.label4.setFont(QFont("华文新魏", 20))
        self.label_REC.move(30,340)

        self.label_VER = QtWidgets.QLabel(self)
        self.label_VER.resize(150, 15)
        self.label5.setFont(QFont("华文新魏", 20))
        self.label_VER.move(200, 40)'''

        self.btn1 = QPushButton("注册", self)
        self.btn1.move(790, 280)
        self.btn1.resize(80,40)
        self.btn2 = QPushButton("识别", self)
        self.btn2.move(1030, 280)
        self.btn2.resize(80,40)
        self.btn = QPushButton("打开摄像头", self)
        self.btn.move(790, 600)
        self.btn.resize(80,40)

        self.btn.clicked.connect(self.show_img)
        self.btn1.clicked.connect(self.rec)
        self.btn2.clicked.connect(self.ver)



        self.le_left = QLineEdit(self)
        self.le_left.move(730, 340)
        self.le_left.resize(200,40)

        self.le_right = QtWidgets.QTextEdit(self)
        self.le_right.move(970, 340)
        self.le_right.resize(200,40)

        #self.palette1 = QtGui.QPalette(self)
        #palette1.setBrush(self.backgroundRole(), QBrush(QPixmap('C:/Users/ZhangLi/Desktor/demo.jpg')))   # 设置背景图片
        #self.palette1.setColor(self.backgroundRole(), QColor(192,253,123))   # 设置背景颜色
        #self.setPalette(palette1)
        self.setGeometry(0, 50, 1400, 800)
        self.setWindowTitle('Face Recognition by ZhangLi')
        self.setStyleSheet('QMainWindow{background-color:rgb(180,205,205)}'
                           'QPushButton{background-color:rgb(139,121,94)}'
                           'QLabel{color:rgb(255,62,150)}')
        #self.setStyleSheet("background-image:url(c:/Users/ZhangLi/Desktor/demo.jpg)")
        self.show()
    def show_img(self):
        global face_rect
        global cam
        # 一个死循环，用来不间断的显示图片
        while True:
            img = cv.QueryFrame(cam)  # 取出视频中的一帧
            # 保存三通道的图片
            src = cv.CreateImage((img.width, img.height), 8, 3)
            cv.Resize(img, src, cv.CV_INTER_LINEAR)
            # 保存灰度图片
            gray = cv.CreateImage((img.width, img.height), 8, 1)
            cv.CvtColor(img, gray, cv.CV_BGR2GRAY)  # 将rgb图片变成灰度图
            cv.EqualizeHist(gray, gray)  # 对灰度图进行直方图均衡化
            rects = detect(gray, cascade)  # 传入图片和分类器，如果检测到人脸，返回人脸的坐标和大小
            face_rect = rects
            # 话那个绿色的人脸框
            draw_rects(src, rects, (0, 255, 0))
            # 显示画框的人脸
            cv.ShowImage('DeepFace ZhangLi', src)
            #path = 'C:/Users/ZhangLi/Desktop/demo.jpg'
            #showImage = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
            #self.show_video.setPixmap(QtGui.QPixmap.fromImage(showImage))
            #png = QPixmap(path)
            #self.show_video.setPixmap(png)
            cv2.waitKey(5) == 27
        cv2.destroyAllWindows()

    def rec(self):
        text, ok = QInputDialog.getText(self, "Input Dialog", "Enter your ID:")
        if ok:
            self.le_left.setText(str(text))
        reg_id = str(text)
        pattern = 1
        print '进入注册函数'
        print '进入注册'
        if pattern == 1:
            tag = 0
            reg_path = './regist_pic'
            # 判断用户是否已经注册
            dir_rec = os.listdir(reg_path)
            for subdir in dir_rec:
                if (subdir == reg_id):  # 说明该用户已经注册
                    print '该用户已经注册!!!!!!\n'
                    tag = 1
            # 该用户未注册
            if tag == 0:
                # 生成该用户的文件夹和注册图片
                os.mkdir(reg_path + '/' + reg_id)
                num = -2
                # 注册五张人脸
                while num < 1:
                    if face_rect:
                    #if True:
                        num = num + 1
                        if num >= 0:
                            register_path = reg_path + '/' + str(reg_id) + '/' + str(num) + '.bmp'
                            register(register_path, cv.QueryFrame(cam), face_rect)
                            print 'now is ' + str(num) + '........\n'
                            time.sleep(0.5)
                self.le_left.setText('注册成功\n' + '他的ID是:' + str(reg_id))
#                self.label4.setText('注册成功')
                path = reg_path + '/' + str(reg_id) + '/' + '0.bmp'
                png = QPixmap(path)
                self.img1.setPixmap(png)
    def ver(self):
        pattern = 2
        self.le_right.setText('请稍后...')
        if pattern == 2:
            print '请稍后...'
            ret = 0
            # ID = 0
            root_path = './regist_pic'
            file_list = loadInfo(root_path)
            # print(file_list)
            print ('#########')
            for md in file_list:
                # 把捕捉到的图片与注册的图片比较
                result = recog(md, cv.QueryFrame(cam))
                print(result)
                if ret < result:
                    ret = result
                    ID = md
            print ret
            print '识别成功!!!!\n' + '他的ID是:' + str(ID)
            print('离开注册函数')
            self.le_right.setText('识别成功!!!!\n' + '他的ID是:' + str(ID))
 #           self.label5.setText('识别成功!!!!')
            self.label1.setText('姓名：' + str(ID))
            self.label2.setText('职业:学生')
            self.label3.setText('签到:已签到')
            path = root_path + '/' + str(ID) + '/' + '0.bmp'
            png = QPixmap(path)
            self.img1.setPixmap(png)
            self.img2.setPixmap(png)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    #t1.start()
    sys.exit(app.exec_())
