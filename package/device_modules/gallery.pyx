from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class ImagePopup(QLabel):
    """ 
    The ImagePopup class is a QLabel that displays a popup, zoomed image 
    on top of another label.  
    """
    def __init__(self, parent):
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
        

class ImageLabel(QLabel):
    """ This widget displays an ImagePopup when the mouse enter its region """
    def enterEvent(self, event:QEvent):
        self.p = ImagePopup(self)
        self.p.show()
        event.accept() 