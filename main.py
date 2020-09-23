# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'HW1_main.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow,QWidget
import sys
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import network_train as network

#global i
i = 0

## Class for sub window needed by 1.4
class Form_14(object):
    def showimg(self):
        img2 = cv2.imread("images\dog.bmp")
        img1 = cv2.flip(img2, 1)
        dst = cv2.addWeighted(img1, 1 - (self.get_weighted.value()/100), img2, self.get_weighted.value()/100, 0)
        image = QtGui.QImage(dst, dst.shape[1],\
                            dst.shape[0], dst.shape[1] * 3,QtGui.QImage.Format_RGB888).rgbSwapped()
        pix = QtGui.QPixmap(image)
        self.value.setText(str(self.get_weighted.value()))
        self.img_load.setPixmap(pix)

    def setupUi(self, Form_14):
        Form_14.setObjectName("Form_14")
        Form_14.resize(621, 490)
        self.centralwidget = QtWidgets.QWidget(Form_14)
        self.centralwidget.setObjectName("centralwidget")
        self.get_weighted = QtWidgets.QSlider(self.centralwidget)
        self.get_weighted.setGeometry(QtCore.QRect(110, 10, 160, 22))
        self.get_weighted.setOrientation(QtCore.Qt.Horizontal)
        self.get_weighted.setObjectName("get_weighted")
        self.get_weighted.setRange(0, 100)
        self.get_weighted.setSingleStep(1)
        self.get_weighted.valueChanged.connect(self.showimg)

        self.text = QtWidgets.QLabel(self.centralwidget)
        self.text.setGeometry(QtCore.QRect(20, 10, 58, 15))
        self.text.setObjectName("text")
        self.text.setText("BLEND: ")

        self.value = QtWidgets.QLabel(self.centralwidget)
        self.value.setGeometry(QtCore.QRect(80, 10, 21, 16))
        self.value.setObjectName("value")
        self.value.setText(str(self.get_weighted.value()))

        self.img_load = QtWidgets.QLabel(self.centralwidget)
        self.img_load.setGeometry(QtCore.QRect(50, 60, 471, 351))
        self.img_load.setObjectName("img_load")
        img = cv2.flip(cv2.imread("images\dog.bmp"), 1)
        img = QtGui.QImage(img, img.shape[1],\
                            img.shape[0], img.shape[1] * 3,QtGui.QImage.Format_RGB888).rgbSwapped()
        img = QtGui.QPixmap(img)
        self.img_load.setPixmap(img)

        Form_14.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Form_14)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 621, 25))
        self.menubar.setObjectName("menubar")
        Form_14.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Form_14)
        self.statusbar.setObjectName("statusbar")
        Form_14.setStatusBar(self.statusbar)

        self.retranslateUi(Form_14)
        QtCore.QMetaObject.connectSlotsByName(Form_14)

    def retranslateUi(self, Form_14):
        _translate = QtCore.QCoreApplication.translate
        Form_14.setWindowTitle(_translate("Form_14", "Blending"))
        

## Funcation call
# Mouse click event
def OnMouseAction(event, x, y, flags, param):
    global x1, y1, i
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1 = x, y
        print(x1, y1)
        if (i == 0):
            global Perspective_array
            Perspective_array = np.float32([[x1, y1]])
        else:
            swap_array = np.float32([[x1, y1]])
            Perspective_array = np.append(Perspective_array, swap_array, axis = 0)
        i += 1
        if i == 4:
            Perspective_translate()

# 3.2 Perspective translate
def Perspective_translate():
    pts1 = Perspective_array
    pts2 = np.float32([[20,20],[450,20],[450,450],[20,450]])

    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img32_origin ,M ,(450,450))
    cv2.imshow('Transfer img', dst)

# 4.2 ~ 4.4 Sobel and magnitude 
def sobel(obj):
    d = cv2.imread('images\School.jpg', cv2.IMREAD_GRAYSCALE)
    sp = d.shape
    height = sp[0]
    weight = sp[1]
    sx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    dSobel = np.zeros((height,weight))
    dSobelx = np.zeros((height,weight))
    dSobely = np.zeros((height,weight))
    Gx = np.zeros(d.shape)
    Gy = np.zeros(d.shape)
    for i in range(height-2):
        for j in range(weight-2):
            Gx[i + 1, j + 1] = abs(np.sum(d[i:i + 3, j:j + 3] * sx))
            Gy[i + 1, j + 1] = abs(np.sum(d[i:i + 3, j:j + 3] * sy))
            dSobel[i+1, j+1] = (Gx[i+1, j+1]*Gx[i+1,j+1] + Gy[i+1, j+1]*Gy[i+1,j+1])**0.5
            dSobelx[i+1, j+1] = np.sqrt(Gx[i+1, j+1])
            dSobely[i + 1, j + 1] = np.sqrt(Gy[i + 1, j + 1])
    if (obj == "x"):
        plt.imshow(dSobelx, cmap= 'gray')
        plt.axis("off")
        plt.show()
    elif (obj == "y"):
        plt.imshow(dSobely, cmap= 'gray')
        plt.axis("off")
        plt.show()
    elif (obj == "norm"):
        norm_x = normalize(dSobelx, 0, 255)
        norm_y = normalize(dSobely, 0, 255)

        fig, axis = plt.subplots(2, 1)
        axis[0].imshow(norm_x, cmap="gray")
        axis[0].axis('off')
        axis[0].set_title("Normalize sobel x")

        axis[1].imshow(norm_y, cmap="gray")
        axis[1].axis('off')
        axis[1].set_title("Normalize sobel y")
        plt.show()

        shape = norm_x.shape
        magnitude = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                magnitude[i][j] = ((norm_x[i][j])**2 + (norm_y[i][j])**2)**(0.5)

        plt.imshow(magnitude, cmap="gray")
        plt.title("Magnitude")
        plt.axis("off")
        plt.show()
# 4.4 Normalization Figure
def normalize(src, a, b):
    shape = src.shape
    dst = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            dst[i][j] = (src[i][j] - src.min()) * abs(b - a) / (src.max() - src.min()) + a
    return dst

class Ui_MainWindow(object):
    def showImage(self):
        img11 = cv2.imread('images\dog.bmp')
        size = img11.shape
        print('Hight =', size[0])
        print('Width =', size[1])
        cv2.imshow('dog', img11)

        # 1.2 color conver
    def colorConver(self):
        img_bgr = cv2.imread('images\color.png')
        cv2.imshow('img_bgr', img_bgr)
        (B, G, R) = cv2.split(img_bgr)
        cv2.imshow('img_rbg',  cv2.merge([G, R, B]))

        # 1.3 img flip
    def imgFlip(self):
        img_flip = cv2.imread('images\dog.bmp')
        cv2.imshow('Before flip', img_flip)
        img_fliped = cv2.flip(img_flip, 1)
        cv2.imshow('After flip', img_fliped)

        # 1.4
        # Unfinished
    #def form14(self):
        

        # 2.1 Global Threshold
    def globalThreshold(self, MainWindow):
        img21 = cv2.imread('images\QR.png', 0)
        cv2.imshow('Original QR code', img21)
        ret, img21_globalTrans = cv2.threshold(img21, 80, 255, cv2.THRESH_BINARY)
        cv2.imshow('Global Threshold', img21_globalTrans)

        # 2.2 Global threshold
    def localThreshold(self, MainWindow):
        img22 = cv2.imread('D:\Engineering Science\OpenCV\Homework\HW1\images\QR.png', 0)
        cv2.imshow('Original QR code', img22)
        img22_localTrans = cv2.adaptiveThreshold(img22, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19, -1)
        cv2.imshow('Local Threshold', img22_localTrans)

        # 3.1 Rotation Scale Transform
    def Transform31(self, MainWindow):
        # show original img
        img31_origin = cv2.imread('D:\Engineering Science\OpenCV\Homework\HW1\images\OriginalTransform.png')
        size = img31_origin.shape
        ##print("origin shape", size)
        cv2.imshow('Original Figure', img31_origin)
        angle = float(self.get_angle.text())
        
        scale = float(self.get_scale.text())
        tx = int(self.get_tx.text())
        ty = int(self.get_ty.text())
        rows, cols = img31_origin.shape[:2]

        M = cv2.getRotationMatrix2D((tx, ty), angle, scale)
        img31_trans = cv2.warpAffine(img31_origin, M, (rows, cols))
        img31_new = cv2.resize(img31_trans, (490, 698),interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Translation Figure', img31_new)
        ##print("trans shape: ", img31_new.shape)

        # 3.2 Perspective trans
    def persTran(self, MainWindow):
        global img32_origin
        img32_origin = cv2.imread('D:\Engineering Science\OpenCV\Homework\HW1\images\OriginalPerspective.png')
        cv2.namedWindow('Perspective_trans')
        # mouse callback event
        #cv2.setMouseCallback('Perspective_trans', OnMouseAction)
        cv2.setMouseCallback('Perspective_trans', OnMouseAction)
        cv2.imshow('Perspective_trans', img32_origin)

        # 4.1
    def Gaussian(self):
        def generate_dst(srcImg):
            m = srcImg.shape[0]
            n = srcImg.shape[1]
            n_channel = 1

            dstImg = np.zeros((m - gaussian_kernel.shape[0]+1, n - gaussian_kernel.shape[0]+1,n_channel ))
            return dstImg
        def conv_2d(src,kernel,k_size):
            dst = generate_dst(src)
            conv(src,dst,kernel,k_size)

            return dst
        def conv(src, dst, kernal, k_size):
            for i in range(dst.shape[0]):
                for j in range(dst.shape[1]):
                    value = _con_each(src[i:i+k_size,j:j+k_size],kernal)
                    dst[i,j] = value
        def _con_each(src_block,kernel):
            pixel_count = kernel.size
            pixel_sum = 0
            _src = src_block.flatten()
            _kernel = kernel.flatten()

            for i in range(pixel_count):
                pixel_sum += _src[i]*_kernel[i]

            value = pixel_sum / pixel_count

            if (value < 0):
                value = 0
            elif(value > 255):
                value = 255

            return value
        def test_conv(src,kernel,k_size):
            dst = conv_2d(src,kernel,k_size)
            dst = dst.astype(int)
            dst = np.squeeze(dst)

            plt.imshow(dst, cmap = 'gray')
            
            plt.axis("off")
            plt.show()

        ## 3*3 Gassian filter
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x**2+y**2))
        # Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

        # get src img
        src = cv2.imread('images\School.jpg', cv2.IMREAD_GRAYSCALE)

        test_conv(src,gaussian_kernel,3)

        # 4.2 to 4.4
    def sobel_x(self):
        sobel("x")

    def sobel_y(self):
        sobel("y")

    def func44(self):
        sobel("norm")

        # 5.1 Show several training image
    def rdImg(self):
        network.get_trainImg(False)

        # 5.2 show Hypreparameter setting
    def get_hyperparm(self):
        print("hyperparameters: ")
        print("batch size: 32")
        print("learning rate: 0.001")
        print("optimizer: SGD")
    
        # 5.3 Train for one epoch
    def one_epoch(self):
        network.LeNet_5(1, "iteration")

        # 5.4  train for 50 epoch, and keep the model
    def train(self):
        network.LeNet_5(50, "epoch")

        # 5.5 predict
    def func55(self, MainWindow):
        index = int(self.getImageIndex.text())
        #print(index)
        predict = network.Predict()
        predict.predict(index)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1108, 580)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(10, 10, 181, 281))
        self.textBrowser.setObjectName("textBrowser")
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_2.setGeometry(QtCore.QRect(210, 10, 231, 161))
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.textBrowser_3 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_3.setGeometry(QtCore.QRect(460, 10, 301, 441))
        self.textBrowser_3.setObjectName("textBrowser_3")
        self.textBrowser_4 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_4.setGeometry(QtCore.QRect(480, 40, 256, 331))
        self.textBrowser_4.setObjectName("textBrowser_4")
        self.textBrowser_5 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_5.setGeometry(QtCore.QRect(500, 70, 221, 221))
        self.textBrowser_5.setObjectName("textBrowser_5")
        self.pushButton_11 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_11.setGeometry(QtCore.QRect(30, 50, 131, 41))
        self.pushButton_11.setObjectName("pushButton_11")
        self.pushButton_11.clicked.connect(self.showImage)

        self.pushButton_12 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_12.setGeometry(QtCore.QRect(30, 110, 131, 41))
        self.pushButton_12.setObjectName("pushButton_12")
        self.pushButton_12.clicked.connect(self.colorConver)

        self.pushButton_13 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_13.setGeometry(QtCore.QRect(30, 170, 131, 41))
        self.pushButton_13.setObjectName("pushButton_13")
        self.pushButton_13.clicked.connect(self.imgFlip)

        self.pushButton_14 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_14.setGeometry(QtCore.QRect(30, 230, 131, 41))
        self.pushButton_14.setAutoFillBackground(False)
        self.pushButton_14.setObjectName("pushButton_14")
        #self.pushButton_14.clicked.connect(self.form14)

        self.textBrowser_6 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_6.setGeometry(QtCore.QRect(210, 190, 231, 271))
        self.textBrowser_6.setObjectName("textBrowser_6")
        self.pushButton_21 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_21.setGeometry(QtCore.QRect(232, 37, 171, 51))
        self.pushButton_21.setObjectName("pushButton_21")
        self.pushButton_21.clicked.connect(self.globalThreshold)

        self.pushButton_22 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_22.setGeometry(QtCore.QRect(230, 100, 171, 51))
        self.pushButton_22.setObjectName("pushButton_22")
        self.pushButton_22.clicked.connect(self.localThreshold)

        self.pushButton_41 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_41.setGeometry(QtCore.QRect(230, 220, 181, 41))
        self.pushButton_41.setObjectName("pushButton_41")
        self.pushButton_41.clicked.connect(self.Gaussian)

        self.pushButton_42 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_42.setGeometry(QtCore.QRect(230, 280, 181, 41))
        self.pushButton_42.setObjectName("pushButton_42")
        self.pushButton_42.clicked.connect(self.sobel_x)

        self.pushButton_43 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_43.setGeometry(QtCore.QRect(230, 340, 181, 41))
        self.pushButton_43.setObjectName("pushButton_43")
        self.pushButton_43.clicked.connect(self.sobel_y)

        self.pushButton_44 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_44.setGeometry(QtCore.QRect(230, 400, 181, 41))
        self.pushButton_44.setObjectName("pushButton_44")
        self.pushButton_44.clicked.connect(self.func44)

        self.get_angle = QtWidgets.QLineEdit(self.centralwidget)
        self.get_angle.setGeometry(QtCore.QRect(550, 100, 71, 31))
        self.get_angle.setObjectName("get_angle")

        self.get_scale = QtWidgets.QLineEdit(self.centralwidget)
        self.get_scale.setGeometry(QtCore.QRect(550, 150, 71, 31))
        self.get_scale.setObjectName("get_scale")

        self.get_tx = QtWidgets.QLineEdit(self.centralwidget)
        self.get_tx.setGeometry(QtCore.QRect(550, 190, 71, 31))
        self.get_tx.setObjectName("get_tx")

        self.get_ty = QtWidgets.QLineEdit(self.centralwidget)
        self.get_ty.setGeometry(QtCore.QRect(550, 240, 71, 31))
        self.get_ty.setObjectName("get_ty")

        self.pushButton_31 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_31.setGeometry(QtCore.QRect(500, 310, 221, 51))
        self.pushButton_31.setObjectName("pushButton_31")
        self.pushButton_31.clicked.connect(self.Transform31)

        self.pushButton_32 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_32.setGeometry(QtCore.QRect(500, 380, 221, 51))
        self.pushButton_32.setObjectName("pushButton_32")
        self.pushButton_32.clicked.connect(self.persTran)

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(510, 100, 41, 31))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(510, 150, 31, 21))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(520, 200, 21, 21))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(520, 240, 21, 21))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(630, 100, 21, 31))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(630, 190, 41, 21))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(630, 240, 41, 21))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.textBrowser_7 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_7.setGeometry(QtCore.QRect(770, 10, 291, 321))
        self.textBrowser_7.setObjectName("textBrowser_7")
        self.pushButton_51 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_51.setGeometry(QtCore.QRect(800, 70, 241, 31))
        self.pushButton_51.setObjectName("pushButton")
        self.pushButton_51.clicked.connect(self.rdImg)

        self.pushButton_52 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_52.setGeometry(QtCore.QRect(800, 110, 241, 31))
        self.pushButton_52.setObjectName("pushButton_2")
        self.pushButton_52.clicked.connect(self.get_hyperparm)

        self.pushButton_53 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_53.setGeometry(QtCore.QRect(800, 150, 241, 31))
        self.pushButton_53.setObjectName("pushButton_3")
        self.pushButton_53.clicked.connect(self.one_epoch)

        self.pushButton_54 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_54.setGeometry(QtCore.QRect(800, 200, 241, 31))
        self.pushButton_54.setObjectName("pushButton_4")
        self.pushButton_54.clicked.connect(self.train)

        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(800, 240, 101, 21))
        self.label_5.setObjectName("label_5")

        self.getImageIndex = QtWidgets.QLineEdit(self.centralwidget)
        self.getImageIndex.setGeometry(QtCore.QRect(910, 240, 141, 21))
        self.getImageIndex.setObjectName("lineEdit")
        self.getImageIndex.setObjectName("getImageIndex")

        self.pushButton_55 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_55.setGeometry(QtCore.QRect(800, 280, 241, 31))
        self.pushButton_55.setObjectName("pushButton_5")
        self.pushButton_55.clicked.connect(self.func55)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1108, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.textBrowser.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt; font-weight:600;\">1. Image Processing</span></p></body></html>"))
        self.textBrowser_2.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt; font-weight:600;\">2. Adaptive Threshold</span></p></body></html>"))
        self.textBrowser_3.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt; font-weight:600;\">3. Image Transformation</span></p></body></html>"))
        self.textBrowser_4.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt; font-weight:600;\">3.1 Rot, scale, Translate</span></p></body></html>"))
        self.textBrowser_5.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt; font-weight:600;\">Parameters</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:10pt; font-weight:600;\"><br /></p></body></html>"))
        self.pushButton_11.setText(_translate("MainWindow", "1.1 Load Image"))
        self.pushButton_12.setText(_translate("MainWindow", "1.2 Color Conversion"))
        self.pushButton_13.setText(_translate("MainWindow", "1.3 Image Flipping"))
        self.pushButton_14.setText(_translate("MainWindow", "1.4 Blending"))
        self.textBrowser_6.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt; font-weight:600;\">4. Convolution</span></p></body></html>"))
        self.pushButton_21.setText(_translate("MainWindow", "2.1 Global Threshold"))
        self.pushButton_22.setText(_translate("MainWindow", "2.2 Local Threshold"))
        self.pushButton_41.setText(_translate("MainWindow", "4.1 Gaussian"))
        self.pushButton_42.setText(_translate("MainWindow", "4.2 Sobel X"))
        self.pushButton_43.setText(_translate("MainWindow", "4.3 Sobel Y"))
        self.pushButton_44.setText(_translate("MainWindow", "4.4 Magnitude"))
        self.pushButton_31.setText(_translate("MainWindow", "3.1 Rotation, scaling, translation"))
        self.pushButton_32.setText(_translate("MainWindow", "3.2 Perspective Transform"))
        self.label.setText(_translate("MainWindow", "Angle: "))
        self.label_2.setText(_translate("MainWindow", "Scale:"))
        self.label_3.setText(_translate("MainWindow", "Tx:"))
        self.label_4.setText(_translate("MainWindow", "Ty:"))
        self.label_6.setText(_translate("MainWindow", "deg"))
        self.label_7.setText(_translate("MainWindow", "pixel"))
        self.label_8.setText(_translate("MainWindow", "pixel"))
        self.textBrowser_7.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:500; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:100px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt; font-weight:600;\">5.0 Training MNIST Classifier Using LeNet5</span></p></body></html>"))
        self.pushButton_51.setText(_translate("MainWindow", "5.1 Show Train Images"))
        self.pushButton_52.setText(_translate("MainWindow", "5.2 Show Hyperparameters"))
        self.pushButton_53.setText(_translate("MainWindow", "5.3 Train 1 Epoch"))
        self.pushButton_54.setText(_translate("MainWindow", "5.4 Show Training Result"))
        self.pushButton_55.setText(_translate("MainWindow", "5.5 Inference"))
        self.label_5.setText(_translate("MainWindow", "Test Image Index:"))
        self.getImageIndex.setText(_translate("MainWindow", "(0~9999)"))

class MyMain(QMainWindow,Ui_MainWindow): #繼承主視窗函式的類, 繼承編寫的主函式
    def __init__(self):
        super().__init__()
        self.setupUi(self)  # 初始化執行A視窗類下的 setupUi 函式
        

class SubWindow(QMainWindow,Form_14):
    def __init__(self):
        super().__init__()
        self.setupUi(self)  # 初始化執行B視窗類下的 setupUi 函式




if __name__ == "__main__":
    app = QApplication(sys.argv)
    A1 = MyMain()
    B1 = SubWindow() 

    A1.pushButton_14.clicked.connect(B1.show) #Form_14 開啟按鈕

    A1.show()
    sys.exit(app.exec_())


#if __name__ == '__main__':  
#    app = QtWidgets.QApplication(sys.argv)
#    MainWindow = QtWidgets.QMainWindow()
#    ui = Ui_MainWindow()
#    sub = Form_14()
#    ui.setupUi(MainWindow)
#    sub.setupUi(MainWindow)

#    ui.pushButton_14.clicked.connect(sub.show)

#    MainWindow.show()
#    sys.exit(app.exec_())