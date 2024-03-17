from ximea import xiapi
import os,sys,threading,cv2, pathlib, random,cython, logging
import numpy as np, time
from classes import roiNet, pupilNet, rangeNet, refNet###################################
import sensors
from sensors import __camera__, led_module
import _io
# import matplotlib.pyplot as plt
from threading import Thread
from typing import *


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logging.info('__ main file called __')

t:float = time.time()
models = '/home/ibex/Documents/trt_models'

ROINET:roiNet = None#roiNet(os.path.join(models,'roinet_256x128_annotationsiz2.trt'))
PUPILNET:pupilNet = None#pupilNet(os.path.join(models,'deeppupilnet.trt'))
RANGENET:rangeNet = None#rangeNet(os.path.join(models,'rangeNet.trt'))
REFNET:refNet = None#refNet(os.path.join(models,'refnet11_0.1158.trt'))
# REFNET = refNet(os.path.join(models,'refnet2.trt'))

logging.info(f'loading time of all models: {time.time()-t}')

t:float = time.time()
camera:xiapi.Camera
img:xiapi.Image
# os.system('echo ibextech1523 | sudo chmod 666 /dev/ttyTHS1')
cam:__camera__ = __camera__()
camera,img = cam.open_device()
logging.info(f'loading time of camera: {time.time()-t}')

leds:List[int]= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38]
led_board:led_module = led_module(leds)
led_pattern:bytearray = led_board.byte_arrays['initial_pattern']['trigger_on'][0]

@cython.cclass
class infoClass():
    frame:np.ndarray
    centers:Tuple[Tuple[float,float],Tuple[float,float]]
    Range:str


@cython.cclass
class Xcamera():
    serial_port:serial.serialposix.Serial
    frame:np.ndarray
    frames:List[np.ndarray]
    monocular:bool
    centers:Tuple[Tuple[float,float],Tuple[float,float]]
    inrange_occurance:int

    def __cinit__(self, monocular:bool=False)->None:
        self.conf()
        try:
            camera.start_acquisition()
        except:
            pass
        self.serial_port = led_board.initialize()
        self.frame = np.zeros((1024,1280))
        self.monocular = monocular
        self.inrange_occurance = 0

    def outsidemodels(self)->Tuple[pupilNet,refNet]:
        return PUPILNET,REFNET
    
    def conf(self):
        self.centers = ((),())

    @property
    def scale(self)->float:
        return (1000-cam.focal_length_mm)/cam.focal_length_mm

    def round225(self,x:float)->float:
        return np.round(x/0.25)*0.25

    def capture(self):
        idx:int
        i:int
        led:bytearray

        self.serial_port.write(led_board.byte_arrays['all_off_trigger_on'])
        all_bytes:List[bytearray] = led_board.byte_arrays['trigger_on']

        for idx in range(len(all_bytes)):
            all_bytes[idx][6] = led_board.colors_trigger_on[random.randint(0,9)]

        self.frames=[None]*len(all_bytes)
        for i,led in enumerate(all_bytes):
            self.serial_port.write(led)
            camera.get_image(img)
            frame = img.get_image_data_numpy()
            self.frames[i] = frame
        return self.frames

    def get_image(self,crop_limit:int=300)->np.ndarray:
        #print('get_image in main.pyx')
        led_pattern[6] = led_board.colors_trigger_on[random.randint(0,9)]
        self.serial_port.write(led_pattern)
        camera.get_image(img)
        return img.get_image_data_numpy()[crop_limit:,:]

    def get_distance(self,frame)->Dict[str,Union[np.ndarray,str,Tuple[Tuple[float]]]]:
        logging.info(f'frame shape:{frame.shape}')
        info = {'frame':[],'range':'unknown','centers':()}

        self.centers = ROINET.inference(frame)
        #print('________________________',self.centers)
        centerslist = []
        
        if len(self.centers[0])!=0:
            centerslist.append(self.centers[0])
        
        if len(self.centers[1])!=0:
            centerslist.append(self.centers[1])
        
        if not self.monocular and len(centerslist)<2:
            print('here')
            info = {'frame':frame,'range':'unknown','centers':centerslist}
        else:
            for center in centerslist:
                a = tuple(np.subtract(center,50))
                b = tuple(np.add(center,50))
                #frame = cv2.rectangle(frame, a, b, (255)*3, 2)

            Range:str = RANGENET.inference(frame,{'left':(centerslist[0],(),0),'right':(centerslist[1],(),0)})
            frame = cv2.putText(frame, Range, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255)*3, 2, cv2.LINE_AA)
            if Range == 'inrange':
                self.inrange_occurance+=1

            logger_file = open('log.log','w')
            logger_file.write(f'{self.inrange_occurance}_{Range}')
            logger_file.close()
            
            info = {'frame':frame,'range':Range,'centers':centerslist}
            #print('___',info)
        return info

    def get_ref(self,centers, instant_capture=False)->Dict[str,cython.float]:
        logger_file:_io.TextIOWrapper
        frames:List[np.ndarray]
        refractive_errors:Dict[str,Dict[str,cython.float]]
        
        if self.inrange_occurance>3:
            frames = self.capture()
            rois = [PUPILNET.inference(image,centers) for image in frames]
            refractive_errors = REFNET.inference(rois)
            self.inrange_occurance = 0

            logger_file = open('refractive_errors.log','w')
            logger_file.write(f'{refractive_errors}')
            logger_file.close()
        elif instant_capture:
            frames = self.capture()
            rois = [PUPILNET.inference(image,centers) for image in frames]
            refractive_errors = REFNET.inference(rois)
            self.inrange_occurance = 0

            logger_file = open('refractive_errors.log','w')
            logger_file.write(f'{refractive_errors}')
            logger_file.close()
        else:
            refractive_errors = {}
            rois = []
        return refractive_errors, rois