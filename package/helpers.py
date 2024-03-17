from __future__ import print_function
from inspect import currentframe
import os
import time
from typing import *
import numpy as np
import cv2
import imageio as imio

from PyQt5.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QWidget,
    QProgressBar,
)
from PyQt5.QtGui import QMovie
from PyQt5 import QtCore
from mycolorpy import colorlist as mcp
colors = mcp.gen_color(cmap="jet",n=201)[90:]

def get_linenumber()->str:
    # Debugging function, returns line number of where this function is called 
    cf = currentframe()
    return 'line: '+str(cf.f_back.f_lineno)


class PurkinjeReport(QWidget):
    def __init__(self,x_direction:int,y_direction:int):
        super().__init__()
        # calling initUI method
        self.initUI(x_direction,y_direction)
    # method for creating widgets
    def initUI(self,x_direction:int,y_direction:int):
        write_log(f'x_direction:{x_direction}')
        write_log(f'y_direction:{y_direction}')
        write_log(get_linenumber())
        self.layout = QHBoxLayout()
        self.glayout = QGridLayout()
        self.layout.addLayout(self.glayout)
        # self.layout.setFixedSize(100,100)
        #write_log(get_linenumber())
        # creating progress bar
        self.x_direction = x_direction
        self.y_direction = y_direction
        #write_log(get_linenumber())
        #topvertical
        self.vpbartop = QProgressBar(self)
        self.vpbartop.setOrientation(QtCore.Qt.Vertical)
        #write_log(get_linenumber())
        # setting its geometry
        self.vpbartop.setFixedSize(5, 45)
        self.vpbartop.setTextVisible(False)
        self.glayout.addWidget(self.vpbartop,0,5,alignment=QtCore.Qt.AlignCenter)
        #write_log(get_linenumber())
        #bottomvertical
        self.vpbarbottom = QProgressBar(self)
        self.vpbarbottom.setOrientation(QtCore.Qt.Vertical)
        #write_log(get_linenumber())
        # setting its geometry
        self.vpbarbottom.setFixedSize(5, 45)
        self.vpbarbottom.setInvertedAppearance(True)
        self.vpbarbottom.setTextVisible(False)
        self.glayout.addWidget(self.vpbarbottom,10,5,alignment=QtCore.Qt.AlignCenter)
        #write_log(get_linenumber())
        #tophorizontal
        self.hpbarright = QProgressBar(self)
        self.hpbarright.setOrientation(QtCore.Qt.Horizontal)
        #write_log(get_linenumber())
        # setting its geometry
        self.hpbarright.setTextVisible(False)
        self.hpbarright.setFixedSize(45, 5)
        self.glayout.addWidget(self.hpbarright,5,10,alignment=QtCore.Qt.AlignCenter)
        #write_log(get_linenumber())
        #bottomhorizontal
        self.hpbarleft = QProgressBar(self)
        self.hpbarleft.setOrientation(QtCore.Qt.Horizontal)
        #write_log(get_linenumber())
        # setting its geometry
        self.hpbarleft.setFixedSize(45, 5)
        self.hpbarleft.setInvertedAppearance(True)
        self.hpbarleft.setTextVisible(False)
        self.glayout.addWidget(self.hpbarleft,5,0,alignment=QtCore.Qt.AlignCenter)
        #write_log(get_linenumber())
        self.fill_value()
        #write_log(get_linenumber())
        self.setLayout(self.glayout)
        #write_log(get_linenumber())
        self.show()


    def fill_value(self):
        # setting for loop to set value of progress bar
        if self.y_direction>0:
            write_log(get_linenumber())
            for i in range(self.y_direction+1):
                # slowing down the loop
                time.sleep(0.005)
                color = colors[i]
                style = '"QProgressBar""{""background-color : white;""border : 2px""}""QProgressBar::chunk ""{""background-color: '+color+';""}"'
                self.vpbartop.setStyleSheet(eval(style))
                self.vpbarbottom.setStyleSheet(eval(style))
                self.hpbarright.setStyleSheet(eval(style))
                self.hpbarleft.setStyleSheet(eval(style))
                self.vpbartop.setValue(i)
            write_log(get_linenumber())
        else:
            write_log(get_linenumber())
            write_log(f'-1*self.y_direction:{-1*self.y_direction}')
            y_direction = -1*self.y_direction
            for i in range(y_direction+1):
                # slowing down the loop
                time.sleep(0.005)
                color = colors[i]
                style = '"QProgressBar""{""background-color : white;""border : 2px""}""QProgressBar::chunk ""{""background-color: '+color+';""}"'
                self.vpbartop.setStyleSheet(eval(style))
                self.vpbarbottom.setStyleSheet(eval(style))
                self.hpbarright.setStyleSheet(eval(style))
                self.hpbarleft.setStyleSheet(eval(style))
                self.vpbarbottom.setValue(i)
            write_log(get_linenumber())

        if self.x_direction>0:
            write_log(get_linenumber())
            for i in range(self.x_direction+1):
                # slowing down the loop
                time.sleep(0.005)
                color = colors[i]
                style = '"QProgressBar""{""background-color : white;""border : 2px""}""QProgressBar::chunk ""{""background-color: '+color+';""}"'
                self.vpbartop.setStyleSheet(eval(style))
                self.vpbarbottom.setStyleSheet(eval(style))
                self.hpbarright.setStyleSheet(eval(style))
                self.hpbarleft.setStyleSheet(eval(style))
                self.hpbarright.setValue(i)
            write_log(get_linenumber())
        else:
            write_log(get_linenumber())
            x_direction = -1*self.x_direction
            for i in range(x_direction+1):
                # slowing down the loop
                time.sleep(0.005)
                color = colors[i]
                style = '"QProgressBar""{""background-color : white;""border : 2px""}""QProgressBar::chunk ""{""background-color: '+color+';""}"'
                self.vpbartop.setStyleSheet(eval(style))
                self.vpbarbottom.setStyleSheet(eval(style))
                self.hpbarright.setStyleSheet(eval(style))
                self.hpbarleft.setStyleSheet(eval(style))
                self.hpbarleft.setValue(i)
            write_log(get_linenumber())
