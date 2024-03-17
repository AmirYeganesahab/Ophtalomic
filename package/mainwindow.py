from . import *
from typing import *
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QWidget,
    QPushButton,
    QProgressBar,
    QDialog,
    QMessageBox
)
import cv2
import time
from .helpers.ledHelpers import *
from .helpers.codeHelpers import *
from .configs.read_confs import configs_
configs = configs_()
from .device_modules.sensors import camera_
from .device_modules.sensors import led_module

logging.basicConfig(level=os.environ.get("LOGLEVEL", logging.WARNING))
logging.info('__ classes file called __')


camera_conf, device_conf, ledboard_conf = configs.camera_conf, configs.device_conf, configs.ledboard_conf

os.environ["QT_FONT_DPI"] = "96" # FIX Problem for High DPI and Scale above 100%
# from device_modules.main import *
pg.setConfigOption('exitCleanup', True)
pg.setConfigOption('useWeave',True)

# SET AS GLOBAL
width = int(camera_conf['width'] * device_conf['scale_percent'] / 100)
height = int(camera_conf['height'] * device_conf['scale_percent'] / 100)
cam:camera_ = camera_(exposure_time = camera_conf['exposure_time'],
                    trigger_source = camera_conf['trigger_source'],
                    animation_number = ledboard_conf['animation_number'],
                    animation_speed = ledboard_conf['animation_speed'],
                    intensity = ledboard_conf['intensity'],
                    trigger_delay = ledboard_conf['trigger_delay'],
                    led_delay = ledboard_conf['led_delay'],
                    ledboard_timeout = ledboard_conf['ledboard_timeout'])

camera:xiapi.Camera
img:xiapi.Image
camera,img = cam.open_device()
led_board:led_module = led_module()
serial_port = led_board.open()
ret = led_board.illuminate()
print(ret)


class MainWindow(QMainWindow):
    apply_process:bool = False
    cekiyor:bool = False
    monocular:bool = False
    sNum:int = 0
    info:Dict = {'centers':None,'isOk':None}
    frame:np.ndarray = np.zeros((1024,1280))
    left_speed_list:List[float] = [100]*3
    right_speed_list:List[float] = [100]*3
    frames:List[np.ndarray] = [np.ndarray]*55
    Range:str
    close_app:bool=False
    
    def __init__(self,camera,img)->None:
        
        QMainWindow.__init__(self)
        
        #initial parameters
        self.camera=camera
        self.img = img
        #check if camera and led board works fine
        _ = get_fois(self.trigger())
        # [cv2.imwrite(f'imgs/{i}.png',image) for i,image in enumerate(imgs)]  
        
        self.patient_id=read_last_id()
        # set functions to buttons
        self.set_widgets()

    def trigger(self):
        imgs=[int]*55
        leds = [np.ndarray]*55
        write_log('cam check [camera_led_coupler-initiation test]')
        led_board.trigger()
        if self.camera.get_acquisition_status()=='XI_ON':
            self.camera.stop_acquisition()
            time.sleep(0.05)
        self.camera.start_acquisition()
        time.sleep(0.05)
        i=0
        j=0
        t = time.time()
        while True:
            # all_off_(cam=cam,ser=ser,trigstate=1)
            try:
                r = int.from_bytes(serial_port.read(), "big")
                # print(r)
                leds[i]=r# önce 
                imgs[i]=cam.get_image()
                print(r,i)
                # if i!=r:
                #     raise BufferError(f"Synchronization is not OK!!! loop {i} led {r} which r!=i")
                if i >= 54:
                    print('______',time.time()-t)
                    [cv2.imwrite(f'imgs/{i}.png',image) for i,image in enumerate(imgs)]  
                    break
                i+=1
            except Exception as e:
                # cam.close_device()
                # self.camera,self.img = cam.open_device()
                # self.camera.start_acquisition()
                j+=1
                print(e)
                if j>5:
                    raise BufferError("Intellivis says: Camera and led are not synced. check led trigger.\n\
                                        This happens when camera does not receive any trigger from led board ")
        return imgs

   
    def set_widgets(self):
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        global widgets
        widgets = self.ui
        # USE CUSTOM TITLE BAR | USE AS "False" FOR MAC OR LINUX
        # ///////////////////////////////////////////////////////////////
        Settings.ENABLE_CUSTOM_TITLE_BAR = False
        # APP NAME
        # ///////////////////////////////////////////////////////////////
        title:str = "Deep Vision"
        description:str = "AI for Ophtalmology"
        # APPLY TEXTS
        self.setWindowTitle(title)
        widgets.titleRightInfo.setText(description)
        # TOGGLE MENU
        # ///////////////////////////////////////////////////////////////
        widgets.toggleButton.clicked.connect(lambda: UIFunctions.toggleMenu(self, True))
        # SET UI DEFINITIONS
        # ///////////////////////////////////////////////////////////////
        UIFunctions.uiDefinitions(self)
        # QTableWidget PARAMETERS
        # ///////////////////////////////////////////////////////////////
        widgets.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # BUTTONS CLICK
        # ///////////////////////////////////////////////////////////////
        # LEFT MENUS
        widgets.btn_shutter.clicked.connect(self.shutter_clicked)
        widgets.btn_home.clicked.connect(self.buttonClick)
        widgets.btn_widgets.clicked.connect(self.buttonClick)
        widgets.btn_new.clicked.connect(self.buttonClick)
        widgets.btn_save.clicked.connect(self.buttonClick)
        widgets.btn_exit.clicked.connect(self.shutdown)
        # EXTRA LEFT BOX
        def openCloseLeftBox():
            UIFunctions.toggleLeftBox(self, True)
        widgets.toggleLeftBox.clicked.connect(openCloseLeftBox)
        widgets.extraCloseColumnBtn.clicked.connect(openCloseLeftBox)
        # EXTRA RIGHT BOX
        def openCloseRightBox():
            UIFunctions.toggleRightBox(self, True)
        widgets.settingsTopBtn.clicked.connect(openCloseRightBox)
       
        # SET CUSTOM THEME
        # ///////////////////////////////////////////////////////////////
        useCustomTheme:bool = False
        themeFile:str = "themes\py_dracula_light.qss"

        # SET THEME AND HACKS
        if useCustomTheme:
            # LOAD AND APPLY STYLE
            UIFunctions.theme(self, themeFile, True)

            # SET HACKS
            AppFunctions.setThemeHack(self)


        # SET HOME PAGE AND SELECT MENU
        # ///////////////////////////////////////////////////////////////
        widgets.stackedWidget.setCurrentWidget(widgets.home)
        widgets.btn_home.setStyleSheet(UIFunctions.selectMenu(widgets.btn_home.styleSheet()))
        self.showMaximized()
        widgets.btn_home.click()
        # SHOW APP
        # ///////////////////////////////////////////////////////////////
        self.show()

    def shutdown(self):
        dialog = QDialog()
        dialog.setWindowTitle("Shutdown Confirmation")
        dialog.setWindowIcon(QIcon('icon.png'))
        dialog.setFixedSize(400, 200)

        label = QLabel("Do you want to shut down your computer?")
        label.setAlignment(Qt.AlignCenter)

        yes_button = QPushButton("Yes")
        yes_button.clicked.connect(lambda: os.system("shutdown -h now"))

        no_button = QPushButton("No")
        no_button.clicked.connect(dialog.close)

        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(yes_button)
        layout.addWidget(no_button)

        dialog.setLayout(layout)
        dialog.exec_()

    def closeEvent(self, event):
        led_board.flush()
        close = QMessageBox.question(self,
                                        "QUIT",
                                        "Are you sure want to close the device?",
                                        QMessageBox.Yes | QMessageBox.No)
        if close == QMessageBox.Yes:
            self.close_app=True
            serial_port.close()
            try:
                cam.stop_acquisition()
                cam.close_device()
            except:
                pass
            event.accept()
        else:
            ret=led_board.illuminate()
            event.ignore()

    def shutter_clicked(self)->None:
        self.apply_process=True
   
    def buttonClick(self)->None:
        # GET BUTTON CLICKED
        btn:QPushButton = self.sender()
        btnName:str = btn.objectName()
        # SHOW HOME PAGE
        if btnName == "btn_home":            
            widgets.stackedWidget.setCurrentWidget(widgets.home)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))
            self.imshow()

        # SHOW WIDGETS PAGE # sonuçlar
        if btnName == "btn_widgets":  
            try:
                self.construct_report_panel()  
            except Exception as e:
                write_log(get_linenumber())
                write_log(e)
            write_log(get_linenumber())
            try:
                widgets.stackedWidget.addWidget(self.reportWidget)
                write_log(get_linenumber())
                widgets.stackedWidget.setCurrentWidget(self.reportWidget)
                write_log(get_linenumber())
                UIFunctions.resetStyle(self, btnName)
                write_log(get_linenumber())
                btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))
                write_log(get_linenumber())
                
            except Exception as e:
                write_log(get_linenumber())
                write_log(e)

        # SHOW NEW PAGE # roiler
        if btnName == "btn_new":
            self.roiViewerWidget = QWidget()
            write_log(get_linenumber())
            self.gif=gifgenerator(self.rois)
            
            widgets.stackedWidget.addWidget(self.gif)
            widgets.stackedWidget.setCurrentWidget(self.gif) # SET PAGE
            
            UIFunctions.resetStyle(self, btnName) # RESET ANOTHERS BUTTONS SELECTED
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet())) # SELECT MENU
            
        if btnName == "btn_save":
            pass

    def __close_app__(self)->None:
        led_board.flush()
        cam.close_device()
    
    def pupilChecker(self,frame=[])->None:
        # write_log(get_linenumber())
        centerslist = []
        if np.shape(frame)[0] == 0:
            self.centers = self.roiNet.inference(self.frame.copy())
        else:
            self.centers = self.roiNet.inference(frame)
        # write_log(get_linenumber())
        centerslist.append(self.centers[0]) if len(self.centers[0])!=0 else None
        centerslist.append(self.centers[1]) if len(self.centers[1])!=0 else None
        # write_log(get_linenumber())

        # 1/2 pupils found (for monocular 1)
        self.info['isOk'] = (len(centerslist)==1 and len(centerslist[0])==2) if self.monocular else (len(centerslist)==2 and len(centerslist[0])==2 and len(centerslist[1])==2)
        # write_log(f'main.py: {get_linenumber()}\n centers: {str(centerslist)}')
        if self.info['isOk']:
            pupilsAreValid = self.pupils_are_valid(left = centerslist[0],right = centerslist[1])
            motionIsFast = self.fast_motion(left_eye = centerslist[0],right_eye = centerslist[1])
            self.info['centers']=centerslist
        else:
            return
        
        # pupils are valid and patient is still
        self.info['isOk'] = self.info['isOk'] and pupilsAreValid and (not motionIsFast)
        if not self.info['isOk']:
            return
          
    def get_decentrization(self,frame:np.ndarray,ellipse:Tuple[Union[Tuple[float],float]]):
        # purkinje stuff
        # x1,y1 = np.subtract(ellipse[0],np.divide(ellipse[1],2))
        x1,y1 = np.subtract(ellipse[0],(50,50))
        x1 = int(round(x1))
        y1 = int(round(y1))
        
        # x2,y2 = np.add(ellipse[0],np.divide(ellipse[1],2))
        x2,y2 = np.add(ellipse[0],(50,50))
        x2 = int(round(x2))
        y2 = int(round(y2))

        new_roi = frame[y1:y2,x1:x2]
        
        ret,binary = cv2.threshold(new_roi,0.5,1, cv2.THRESH_BINARY)
        mask = np.multiply(new_roi,binary)
        mask = cv2.normalize(mask, mask, 0, 256, cv2.NORM_MINMAX,dtype = cv2.CV_8UC1)
        ret, th = cv2.threshold(mask, 255*0.7,255, cv2.THRESH_BINARY)
        cnts = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x,y = np.mean(np.where(mask==np.max(mask)), axis = 1)
        x = int(round(x))
        y = int(round(y))
        t = 1
        while True:
            filled_purkinje = flood_fill(mask, (x, y), 255,tolerance=t, in_place=False)
            filled_purkinje = cv2.normalize(filled_purkinje, filled_purkinje, 0, 256, cv2.NORM_MINMAX,dtype = cv2.CV_8UC1)
            _,purk = cv2.threshold(filled_purkinje, 254,256, cv2.THRESH_BINARY)
            purk = cv2.normalize(purk, purk, 0, 1, cv2.NORM_MINMAX,dtype = cv2.CV_8UC1)
            n_nonzeros = np.count_nonzero(purk)
            if n_nonzeros>=5: break
            t+=5
        # purkinje center
        x,y = np.nonzero(purk)
        x,y = int(round(x.mean()))+x1,int(round(y.mean()))+y1
        #pupil center
        x_center,y_center = ellipse[0]
        # pixel difference
        dx = (x_center-x)*pixel2mm*scale #mm
        dy = (y_center-y)*pixel2mm*scale #mm
        # normal difference
        r= np.sqrt(dx**2+dy**2) #mm
        x_decentrization = (dx*(11.5-5),dx*(11.5-2)) #degrees (min,max)
        y_decentrization = (dy*(11.5-2),dy*(11.5-1)) #degrees (min, max)
        decentrization = {'horizontal':{'min':dx*(11.5-5),'max':dx*(11.5-2),'average':dx*(23-7)/2},
                          'vertical':{'min':dy*(11.5-2),'max':dy*(11.5-1), 'average':dx*(23-3)/2}}
        # write_log(str(decentrization))
        return decentrization
    
    def fast_motion(self,left_eye,right_eye):
        if self.sNum>=3:self.sNum = 0
        self.left_speed_list[self.sNum] = np.linalg.norm(left_eye)
        self.right_speed_list[self.sNum] = np.linalg.norm(right_eye)
        self.sNum += 1
        a = np.mean(np.subtract(self.left_speed_list[:-1],self.left_speed_list[1:]))
        b = np.mean(np.subtract(self.right_speed_list[:-1],self.right_speed_list[1:]))
        speed = np.mean([a,b])
        # write_log(f'speed:{speed}')
        # write_log(get_linenumber())
        if abs(speed) > 50:#pixels/frame
            # patient should take his/her head stable
            return True
        return False

    def pupils_are_valid(self,
                        left:Tuple[cython.float,cython.float],
                        right:Tuple[cython.float,cython.float], 
                        threshold:cython.int = 20)->bool:
        d:np.ndarray
        pdist:cython.float
        alpha:cython.float
        try:
            d = np.subtract(left,right)
            pdist = np.linalg.norm(d)*pixel2mm*scale #mm
            if 35 > pdist > 100:
                return False
            if d[0]==0:
                return False
            alpha = np.rad2deg(np.arctan(d[1]/d[0]))
            if abs(alpha) > threshold:
                return False
            return True
        except:
            return False
        
    def imshow(self)->None:
        def rangeFetcher()->None:
            # write_log(get_linenumber())
            centerslist = []
            # t = time.time()
            self.centers = self.roiNet.inference(self.frame.copy())
            # write_log('time to run roiNet:'+str(time.time()-t))
            centerslist.append(self.centers[0]) if len(self.centers[0])!=0 else None
            centerslist.append(self.centers[1]) if len(self.centers[1])!=0 else None
            
            # write_log(centerslist)
            if not self.monocular and len(centerslist)<2:
                self.Range = 'unknown'
            else:
                # t = time.time()
                self.Range = self.rangeNet.inference(Oframe=self.frame.copy(),centers={'left':(centerslist[0],(),0),'right':(centerslist[1],(),0)})
                # print(self.Range)
                # self.Range = "inrange"
                # write_log('time to run rangeNet: '+str(time.time()-t))
                if self.Range == 'inrange':
                    self.inrange_occurance+=1 
                else:
                    self.inrange_occurance=0

            self.info = {'range':self.Range,'centers':centerslist}
        
        if self.roiNet is None:
            from nn import roiNet, pupilNet, refNet, rangeNet
            self.roiNet:roiNet
            self.pupilNet:pupilNet
            self.refNet:refNet
            models:str = '/home/ibex/Documents/models2trt/trt'
            self.roiNet = roiNet(path=os.path.join(models,'roiNet.trt'))
            self.pupilNet = pupilNet(path=os.path.join(models,'pupilNet.trt'))
            self.refNet = refNet(path=os.path.join(models,'new_refnet11_0.1158_oppset13.trt'),multiplier=7)
            self.rangeNet = rangeNet(path=os.path.join(models,'rangeNet.trt'))
            #self.process_thread = threading.Thread(target=self.process())
            write_log('models loaded')

        center:Tuple[int]
        e:str
        c:np.ndarray
        
        while True:
            if self.close_app:break
            if len(self.frame)==0: continue
            if self.apply_process:
                # t=time.time()
                fs = self.trigger()
                [cv2.imwrite(f'captures/{self.patient_id}_{i}.png',image) for i,image in enumerate(fs)]  
                imgs:List[np.ndarray] = get_fois(fs)
                # write_log(f'get_fois time: {time.time()-t}')
                # write_log(len(imgs))
                # [cv2.imwrite(f'captures/{self.patient_id}_{i}.png',image) for i,image in enumerate(imgs)]  
                write_last_id(self.patient_id) 
                # print(self.patient_id, type(self.patient_id))
                
                self.patient_id+=1

                self.process(imgs)
                self.apply_process=False

            ret,self.frame = cam.illuminate()
            # print(ret)
            # self.frame= self.get_image(leds=self.leds['3led'])
            t=time.time()
            self.pupilChecker()
            self.display = cv2.merge([self.frame.copy(),self.frame.copy(),self.frame.copy()])
            
            self.display = cv2.putText(self.display, f'Patient ID: {self.patient_id}', (150,100),\
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255)*3, 2, cv2.LINE_AA)
            # self.display = cv2.putText(self.display, self.Range, (150,150),\
            #                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255)*3, 2, cv2.LINE_AA)
            try:
                if self.info['isOk']:
                    self.display[:,:,0]=np.divide(self.display[:,:,0],2)
                else:
                    self.display[:,:,2]=np.divide(self.display[:,:,2],2)
                
            except Exception as e:
                write_log(e)
            
            if self.cekiyor:
                self.display = cv2.putText(self.display, 'Acquiring', (350,100),\
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255)*3, 2, cv2.LINE_AA)
            
            self.display = cv2.putText(self.display, str(self.patient_id), (-350,100),\
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255)*3, 2, cv2.LINE_AA)

            qimg:QPixmap = self.convert_cv_qt(self.display)
            try:
                widgets.image_label.setPixmap(qimg)
            except Exception as e:
                write_log(f'in {get_linenumber()}:\n{e}')
            cv2.waitKey(1)
            
    def process(self,frames):
        write_log(f"self.apply_process: {get_linenumber()}")
        # print('_______________________________________________')
        self.rois = [self.pupilNet.inference(image,self.info['centers']) for image in frames]
        # print(self.rois)
        # write_log(f"pupilNet: {get_linenumber()}")
        
        self.refractive_errors = self.refNet.inference(self.rois)
        # write_captures('_________________________________')
        write_captures(f'patient_id:{self.patient_id-1}')
        write_captures(f'refractive errors:\n {self.refractive_errors}')
        write_captures('_________________________________')
        # write_log(f"refNet: {get_linenumber()}")
        info:np.ndarray = self.rois[5][1]
        roi:np.ndarray = self.rois[5][0]
        # write_log(f"rois: {get_linenumber()}")
        self.left_decenterization = self.get_decentrization(frames[5],info['left'])
        # write_log(f"left_decenterization: {get_linenumber()}")
        self.right_decenterization = self.get_decentrization(frames[5],info['right'])
        # write_log(f"right_decenterization: {get_linenumber()}")
        self.left_pupilsize = int(round(np.mean(info['left'][1])*pixel2mm*scale))#mm
        # write_log(f"left_pupilsize: {get_linenumber()}")
        self.right_pupilsize = int(round(np.mean(info['right'][1])*pixel2mm*scale))#mm
        # write_log(f"right_pupilsize: {get_linenumber()}")
        self.interpupilary_dist = np.linalg.norm(np.subtract(info['left'][0],info['right'][0]))*pixel2mm*scale#mm
        # write_log(f"interpupilary_dist: {get_linenumber()}")
        widgets.btn_widgets.click()
        self.apply_process=False

    def convert_cv_qt(self, cv_img:np.ndarray)->QPixmap:
        '''Convert from an opencv image to QPixmap'''
        h, w,ch = cv_img.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(cv_img, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(480, 640, Qt.KeepAspectRatio)
        p = convert_to_Qt_format.scaled(int(round(w/2)), int(round(h/2)), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def construct_report_panel(self)->None:
        # write_log(f'construct_report_panel {get_linenumber()}')
        degree_sign:str= u'\N{DEGREE SIGN}'
        self.reportWidget = QWidget()
        self.grid = QGridLayout()
        self.reportWidget.setLayout(self.grid)
        
        mainHBox:QHBoxLayout = QHBoxLayout()
        
        leftgroupbox:QGroupBox = QGroupBox('Left')
        leftgroupbox.setStyleSheet(''' font-size: 12px; ''')
        leftVBox:QVBoxLayout = QVBoxLayout()
        
        # write_log(get_linenumber())
        rightgroupbox:QGroupBox = QGroupBox('Right')
        rightgroupbox.setStyleSheet(''' font-size: 12px; ''')
        rightVBox:QVBoxLayout = QVBoxLayout()
        
        write_log(get_linenumber())
        self.leftLayout = QHBoxLayout()
        
        write_log(get_linenumber())
        self.rightLayout = QHBoxLayout()
        # self.rightLayout.addLayout(rightVBox)
        
        write_log(get_linenumber())
        left_refErrors_gbox = QGroupBox('Ref Errors')
        left_refErrors_gbox.setLayout(leftVBox)

        write_log(get_linenumber())
        right_refErrors_gbox = QGroupBox('Ref Errors')
        right_refErrors_gbox.setLayout(rightVBox)

        write_log(get_linenumber())
        right_decentrizations_values_gbox = QGroupBox('')
        left_decentrizations_gbox = QGroupBox('')
        left_decentrizations_values_gbox = QGroupBox('')
        right_decentrizations_gbox = QGroupBox('')
        
        write_log(get_linenumber())
        self.leftLayout.addWidget(left_decentrizations_gbox)
        self.leftLayout.addWidget(left_refErrors_gbox)
        self.leftLayout.addWidget(left_decentrizations_values_gbox)

        write_log(get_linenumber())
        self.rightLayout.addWidget(right_decentrizations_values_gbox)
        self.rightLayout.addWidget(right_refErrors_gbox)
        self.rightLayout.addWidget(right_decentrizations_gbox)
        
        leftgroupbox.setLayout(self.leftLayout)
        rightgroupbox.setLayout(self.rightLayout)
        write_log(get_linenumber())
        mainHBox.addWidget(rightgroupbox)
        mainHBox.addWidget(leftgroupbox)

        write_log(get_linenumber())
        left_hdecent =int(round(self.left_decenterization['horizontal']['average']))
        
        write_log(get_linenumber())
        left_vdecent =int(round(self.left_decenterization['vertical']['average']))
        
        write_log(get_linenumber())
        left_purk = PurkinjeReport(x_direction = left_hdecent,y_direction= left_vdecent).layout
        write_log(get_linenumber())
        
        left_decentrizations_gbox.setLayout(left_purk)
        write_log(get_linenumber())
        lVBox = QVBoxLayout()
        write_log(get_linenumber())
        lvertical:QLabel = QLabel()
        write_log(get_linenumber())
        lvertical.setStyleSheet(''' font-size: 12px; ''')
        write_log(get_linenumber())
        lvertical.setText(f'V: {left_vdecent} {degree_sign}')
        write_log(get_linenumber())
        # =========================================
        font:QFont = QFont()
        font.setPointSize(24)
        write_log(get_linenumber())
        # =========================================
        lvertical.setFont(font)
        write_log(get_linenumber())
        lVBox.addWidget(lvertical)
        write_log(get_linenumber())
        lhorizontal:QLabel = QLabel()
        write_log(get_linenumber())
        lhorizontal.setStyleSheet(''' font-size: 12px; ''')
        lhorizontal.setText(f'H: {left_hdecent} {degree_sign}')
        #lhorizontal.setFont(font)
        lVBox.addWidget(lhorizontal)
        write_log(get_linenumber())
        left_decentrizations_values_gbox.setLayout(lVBox)
        write_log(get_linenumber())
        
        # =========================================
        left_ds = self.refractive_errors['left']['ds']
        left_dc = self.refractive_errors['left']['dc']
        
        left_ax = self.refractive_errors['left']['ax']
        # write_log(f'construct_report_panel {get_linenumber()}')
        left_pupilsize = self.left_pupilsize
        # write_log(f'construct_report_panel {get_linenumber()}')
        
        # write_log(get_linenumber())
        write_log('left:')
        write_log(f'ds:{left_ds}')
        write_log(f'dc:{left_dc}')
        write_log(f'ax:{left_ax}')
        write_log(f'horizontal decentrization:{left_hdecent}')
        write_log(f'vertival decentrization:{left_vdecent}')

        lds:QLabel = QLabel()
        lds.setStyleSheet(''' font-size: 12px; ''')
        lds.setText('ds:')
        lds.setFont(font)
        
        lds_value:QLabel = QLabel()
        lds_value.setStyleSheet(''' font-size: 12px; ''')
        
        lds_value.setText(str(left_ds))
        lds_value.setFont(font)

        ldsHBox:QHBoxLayout = QHBoxLayout()
        
        ldsHBox.addWidget(lds)
        ldsHBox.addWidget(lds_value)
        leftVBox.addLayout(ldsHBox)

        ldc:QLabel = QLabel()
        ldc.setStyleSheet(''' font-size: 12px; ''')
        ldc.setText('dc:')
        ldc.setFont(font)
        
        ldc_value:QLabel = QLabel()
        ldc_value.setStyleSheet(''' font-size: 12px; ''')

        ldc_value.setText(str(left_dc))
        ldc_value.setFont(font)
        
        ldcHBox:QHBoxLayout = QHBoxLayout()
        ldcHBox.addWidget(ldc)
        ldcHBox.addWidget(ldc_value)
        leftVBox.addLayout(ldcHBox)
        
        laxis:QLabel = QLabel()
        laxis.setStyleSheet(''' font-size: 12px; ''')
        laxis.setText('axis:')
        laxis.setFont(font)
        
        laxis_value:QLabel = QLabel()
        laxis_value.setStyleSheet(''' font-size: 12px; ''')

        laxis_value.setText(f"{left_ax} {degree_sign}")
        laxis_value.setFont(font)

        laxisHBox:QHBoxLayout = QHBoxLayout()
        laxisHBox.addWidget(laxis)
        laxisHBox.addWidget(laxis_value)
        leftVBox.addLayout(laxisHBox)

        lsize:QLabel = QLabel()
        lsize.setStyleSheet(''' font-size: 12px; ''')
        lsize.setText('size:')
        lsize.setFont(font)
        
        lsize_value:QLabel = QLabel()
        lsize_value.setStyleSheet(''' font-size: 12px; ''')
        lsize_value.setText(f'{left_pupilsize} mm')
        lsize_value.setFont(font)

        lsizeHBox:QHBoxLayout = QHBoxLayout()
        lsizeHBox.addWidget(lsize)
        lsizeHBox.addWidget(lsize_value)
        leftVBox.addLayout(lsizeHBox)
        # write_log(get_linenumber())

        # =========================================
        right_ds = self.refractive_errors['right']['ds']
        right_dc = self.refractive_errors['right']['dc']
        right_ax = self.refractive_errors['right']['ax']
        right_pupilsize = self.right_pupilsize
        # write_log(get_linenumber())
        right_hdecent =int(round(self.right_decenterization['horizontal']['average']))
        # write_log(get_linenumber())
        right_vdecent =int(round(self.right_decenterization['vertical']['average']))
        write_log('right:')
        write_log(f'ds:{right_ds}')
        write_log(f'dc:{right_dc}')
        write_log(f'ax:{right_ax}')
        rVBox = QVBoxLayout()
        # write_log(get_linenumber())
        rvertical:QLabel = QLabel()
        # write_log(get_linenumber())
        rvertical.setStyleSheet(''' font-size: 12px; ''')
        # write_log(get_linenumber())
        rvertical.setText(f'V: {right_vdecent} {degree_sign}')
        # write_log(get_linenumber())
        rvertical.setFont(font)
        # write_log(get_linenumber())
        rVBox.addWidget(rvertical)
        # write_log(get_linenumber())
        rhorizontal:QLabel = QLabel()
        # write_log(get_linenumber())
        rhorizontal.setStyleSheet(''' font-size: 12px; ''')
        rhorizontal.setText(f'H: {right_hdecent} {degree_sign}')
        #lhorizontal.setFont(font)
        rVBox.addWidget(rhorizontal)
        # write_log(get_linenumber())
        right_decentrizations_values_gbox.setLayout(rVBox)
        # write_log(get_linenumber())

        rds:QLabel = QLabel()
        rds.setStyleSheet(''' font-size: 12px; ''')
        rds.setText('ds:')
        rds.setFont(font)

        rds_value:QLabel = QLabel()
        rds_value.setStyleSheet(''' font-size: 12px; ''')
        rds_value.setText(str(right_ds))
        rds_value.setFont(font)

        rdsHBox:QHBoxLayout = QHBoxLayout()
        rdsHBox.addWidget(rds)
        rdsHBox.addWidget(rds_value)
        rightVBox.addLayout(rdsHBox)

        rdc:QLabel = QLabel()
        rdc.setStyleSheet(''' font-size: 12px; ''')
        rdc.setText('dc:')
        rdc.setFont(font)

        rdc_value:QLabel = QLabel()
        rdc_value.setStyleSheet(''' font-size: 12px; ''')
        rdc_value.setText(str(right_dc))
        rdc_value.setFont(font)

        rdcHBox:QHBoxLayout = QHBoxLayout()
        rdcHBox.addWidget(rdc)
        rdcHBox.addWidget(rdc_value)
        rightVBox.addLayout(rdcHBox)

        raxis:QLabel = QLabel()
        raxis.setStyleSheet(''' font-size: 12px; ''')
        raxis.setText('axis:')
        raxis.setFont(font)

        raxis_value:QLabel = QLabel()
        raxis_value.setStyleSheet(''' font-size: 12px; ''')
        raxis_value.setText(f"{right_ax} {degree_sign}")
        raxis_value.setFont(font)

        raxisHBox:QHBoxLayout = QHBoxLayout()
        raxisHBox.addWidget(raxis)
        raxisHBox.addWidget(raxis_value)
        rightVBox.addLayout(raxisHBox)

        lsize:QLabel = QLabel()
        lsize.setStyleSheet(''' font-size: 12px; ''')
        lsize.setText('size:')
        lsize.setFont(font)

        rsize_value:QLabel = QLabel()
        rsize_value.setStyleSheet(''' font-size: 12px; ''')
        rsize_value.setText(f'{right_pupilsize} mm')
        rsize_value.setFont(font)

        hldecenter:QLabel = QLabel()
        hldecenter.setStyleSheet(''' font-size: 12px; ''')
        hldecenter.setText(f'H')
        hldecenter.setFont(font)

        vldecenter:QLabel = QLabel()
        vldecenter.setStyleSheet(''' font-size: 12px; ''')
        vldecenter.setText(f'V')
        vldecenter.setFont(font)

        hrdecenter:QLabel = QLabel()
        hrdecenter.setStyleSheet(''' font-size: 12px; ''')
        hrdecenter.setText(f'H')
        hrdecenter.setFont(font)

        vrdecenter:QLabel = QLabel()
        vrdecenter.setStyleSheet(''' font-size: 12px; ''')
        vrdecenter.setText(f'V')
        vrdecenter.setFont(font)
        # =========================================
        rdecenter:QVBoxLayout = QVBoxLayout()
        rdecenter.addWidget(hrdecenter)
        rdecenter.addWidget(vrdecenter)
        # =========================================
        ldecenter:QVBoxLayout = QVBoxLayout()
        ldecenter.addWidget(hldecenter)
        ldecenter.addWidget(vldecenter)
        # =========================================
        rdistHBox:QHBoxLayout = QHBoxLayout()
        rdistHBox.addWidget(lsize)
        rdistHBox.addWidget(rsize_value)
        rightVBox.addLayout(rdistHBox)
        
        right_purk = PurkinjeReport(x_direction = -1*right_hdecent,y_direction= -1*right_vdecent).layout
        # write_log(get_linenumber())
        right_decentrizations_gbox.setLayout(right_purk)
        # write_log(get_linenumber())

        # =========================================
        image_view_left:pg.ImageView = pg.ImageView()
        image_view_left.ui.menuBtn.hide()
        image_view_left.ui.roiBtn.hide()
        image_view_right = pg.ImageView()
        image_view_right.ui.menuBtn.hide()
        image_view_right.ui.roiBtn.hide()
        # =========================================
        roileft:np.ndarray = cv2.rotate(self.rois[5][0]['left'],cv2.ROTATE_90_COUNTERCLOCKWISE)
        roileft:np.ndarray = cv2.flip(roileft, 0)
        image_view_left.setImage(roileft)
        # =========================================
        roiright:np.ndarray = cv2.rotate(self.rois[5][0]['right'],cv2.ROTATE_90_COUNTERCLOCKWISE)
        roiright:np.ndarray = cv2.flip(roiright, 0)
        image_view_right.setImage(roiright)
        # =========================================
        self.grid.addWidget(image_view_right,0,0)
        self.grid.addWidget(image_view_left,0,1)
        # =========================================
        # write_log(self.interpupilary_dist)
        interpupilary_dist = self.interpupilary_dist
        write_log(f'interpupilary_dist:{interpupilary_dist}')

        dist_label:QLabel = QLabel()
        dist_label.setStyleSheet(''' font-size: 12px; ''')
        dist_label.setText(f'|________________ Interpupilary distance:  {round(interpupilary_dist,1)} mm ___________________|')
        dist_label.setFont(font)
        dist_label.setFont(font)
        dist_label.setAlignment(QtCore.Qt.AlignCenter)
        # =========================================
        self.grid.addWidget(dist_label,1,0,1,2)
        self.grid.addWidget(rightgroupbox,2,0)
        self.grid.addWidget(leftgroupbox,2,1)
        write_log('report pannel constructed')

    def resizeEvent(self, event)->None:
        # Update Size Grips
        UIFunctions.resize_grips(self)

    def mousePressEvent(self, event)->None:
        # SET DRAG POS WINDOW
        self.dragPos = event.globalPos()

        # PRINT MOUSE EVENTS
        if event.buttons() == Qt.LeftButton:
            print('Mouse click: LEFT CLICK')
        if event.buttons() == Qt.RightButton:
            print('Mouse click: RIGHT CLICK')