# from ximea import xiapi
from __future__ import print_function
from inspect import currentframe

from package import *





if __name__ == "__main__":

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("icon.ico"))
    window = MainWindow(camera=camera, img=img)
    app.aboutToQuit.connect(window.__close_app__)
    
    window.show()
    
    sys.exit(app.exec_())
