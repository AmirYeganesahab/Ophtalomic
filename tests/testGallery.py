# -*- coding: utf-8-*-

"""
This program is licensed under the BSD license.

Copyright (c) 2009, Marco Dinacci
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, 
are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, 
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, 
      this list of conditions and the following disclaimer in the documentation 
      and/or other materials provided with the distribution.
    * Neither the name of the Dino Interactive nor the names of its contributors 
      may be used to endorse or promote products derived from this software 
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR 
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON 
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class ImageLabel(QLabel):
    """ This widget displays an ImagePopup when the mouse enter its region """

    def enterEvent(self, event):
        self.p = ImagePopup(self)
        self.p.show()
        event.accept() 

class ImagePopup(QLabel):
    """ 
    The ImagePopup class is a QLabel that displays a popup, zoomed image 
    on top of another label.  
    """
    def __init__(self, parent:ImageLabel):
        super(QLabel, self).__init__(parent)
        thumb:QPixmap = parent.pixmap()
        imageSize:QSize = thumb.size()
        imageSize.setWidth(imageSize.width()*2)
        imageSize.setHeight(imageSize.height()*2)
        self.setPixmap(thumb.scaled(imageSize,Qt.KeepAspectRatioByExpanding))
        
        # center the zoomed image on the thumb
        position:QPoint = self.cursor().pos()
        position.setX(position.x() - thumb.size().width())
        position.setY(position.y() - thumb.size().height())
        self.move(position)
        
        # FramelessWindowHint may not work on some window managers on Linux
        # so I force also the flag X11BypassWindowManagerHint
        self.setWindowFlags(Qt.Popup | Qt.WindowStaysOnTopHint 
                            | Qt.FramelessWindowHint 
                            | Qt.X11BypassWindowManagerHint)

    def leaveEvent(self, event:QEvent):
        """ When the mouse leave this widget, destroy it. """
        self.destroy()

class ImageGallery(QDialog):
    
    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent)
        self.setWindowTitle("Image Gallery")
        self.setLayout(QGridLayout(self))
    
    def populate(self, pics, size, imagesPerRow=11, 
                 flags=Qt.KeepAspectRatioByExpanding):
        row = col = 0
        rc = {0:[10,5],1:[9,5],2:[8,5],3:[7,5],4:[6,5],5:[5,5],6:[4,5],7:[3,5],8:[2,5],9:[1,5],10:[0,5],
       		11:[10,10],12:[9,9],13:[8,8],14:[7,7],15:[6,6],16:[4,4],17:[3,3],18:[2,2],19:[1,1],20:[0,0],
       		21:[0,10],22:[1,9],23:[2,8],24:[3,7],25:[4,6],26:[6,4],27:[7,3],28:[8,2],29:[9,1],30:[10,0]}
        for i,pic in enumerate(pics):
            row,col = rc[i]
            label = ImageLabel("")
            pixmap = QPixmap(pic)
            pixmap = pixmap.scaled(size, flags)
            label.setPixmap(pixmap)
            self.layout().addWidget(label, row, col)
            '''
            col +=1
            if col % imagesPerRow == 0:
                row += 1
                col = 0
            '''
if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    pics = [f'/home/ibex/Documents/dummy_data/N852_{i}.png' for i in range(30)]
    #pics = ["img1.png", "img2.png", "img3.gif","img4.png"]*4
    ig = ImageGallery()
    ig.populate(pics, QSize(30,30))
    
    ig.show()
    app.exec_()