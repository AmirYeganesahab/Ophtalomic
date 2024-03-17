from __future__ import print_function
from inspect import currentframe

from ximea import xiapi
import cython, logging,os
from typing import *
import serial
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logging.info('__ sensors file called __')
import random

def get_linenumber()->str:
    cf:currentframe = currentframe()
    return 'line: '+str(cf.f_back.f_lineno)

def write_log(ss)->None:
    with open("/home/ibex/Documents/XCamera_modern.v2.0/log.txt", "a") as fp:
        fp.write(str(ss)+"\n")



@cython.cclass
class __camera__():
    def __cinit__(self):
        pass
    @property
    def focal_length_mm(self):
        return 39.2 #mm
    @property
    def focal_length_pixels(self):
        return 860 #pixels
    @property
    def pixel2mm(self):
        return 0.0048 # 1 pixel corresponds to 0.0048 mm
    
    def open_device(self,exposure_time:cython.int = int(25000/2),trigger_source:str='software')->Tuple[xiapi.Camera,xiapi.Image]:
        device:xiapi.Camera = xiapi.Camera()
        device.open_device()
        device.stop_acquisition()
        if trigger_source!='software':
            device.set_param( 'gpi_selector',"XI_GPI_PORT1")
  
        else:
            device.set_param( 'trigger_source',"XI_TRG_SOFTWARE")

        device.set_param( 'trigger_source',"XI_TRG_EDGE_RISING")  
        # device.get_param( 'trigger_selector',"XI_TRG_SEL_EXPOSURE_START")
        device.get_param( 'trigger_selector',"XI_TRG_FRAME_START")
        device.set_param( 'gpi_mode',"XI_GPI_TRIGGER")
        device.set_param( 'gpo_mode',"XI_GPO_EXPOSURE_ACTIVE")
        device.set_param( 'aeag',0)# or 1
        device.set_param('exposure',exposure_time)
        device.set_param( 'ag_max_limit',10)#db
        device.set_param( 'ae_max_limit',10000)#us
        device.set_param( 'exp_priority',0.5)# maximum 1
        device.set_param( 'aeag_level',50)
        device.set_param( 'LUTEnable',0)
        return device,xiapi.Image()

@cython.cclass
class led_module():
    infrared_intensity:int
    rgb_intensity:int
    random_byteArrays:List[Dict[str,List[bytearray]]]
    def __init__(self,infrared_intensity:int=255,rgb_intensity:int=15)->None:
        print('led board conf ...')
        self.infrared_intensity = infrared_intensity
        self.rgb_intensity = rgb_intensity
        
        # write_log(len(self.random_byteArrays))

    @property
    def random_byteArrays(self)->List[Dict[str,List[bytearray]]]:
        return self.randomByteArrays()

    @property
    def serial_port(self)->serial.Serial:
        return serial.Serial(port="/dev/ttyTHS1", baudrate=115200,timeout=0,write_timeout=0)

    @property
    def update(self)->bytearray:
        i:int
        b:int
        update_:bytearray = bytearray(8)
        for i,b in enumerate([186, 222, 254, 0, 0, 0, 0,105]):
            update_[i]=b
        return update_

    def generateByteArray(self,ledNum:int,r:int,g:int,b:int)->bytearray:
        ba:bytearray = bytearray(8)
        ba[0]=0xba
        ba[1]=0xde
        ba[2] = ledNum
        ba[3] = r
        ba[4] = g
        ba[5] = b
        ba[6] = 0
        ba7=ba[7]
        for a in ba[:6]:
            ba7 +=a
        ba[7]=0xff-ba7%256
        return ba

    @property
    def old_led_arrangement(self)->List[int]:
        m:List[int] = []
        m.extend(self.meridiansNum2LedNum[0])
        m.extend(self.meridiansNum2LedNum[60])
        m.extend(self.meridiansNum2LedNum[120])
        #output:List[bytearray] = [self.generateByteArray(ledNum=ledNum,r=intensity,g=0,b=0) for ledNum in m]
        return m

    @property
    def led_arrangement(self)->List[int]:
        m:List[int] = []
        m.extend(self.meridiansNum2LedNum[0])
        m.extend(self.meridiansNum2LedNum[30])
        m.extend(self.meridiansNum2LedNum[60])
        m.extend(self.meridiansNum2LedNum[90])
        m.extend(self.meridiansNum2LedNum[120])
        m.extend(self.meridiansNum2LedNum[150])
        #output:List[bytearray] = [self.generateByteArray(ledNum=ledNum,r=intensity,g=0,b=0) for ledNum in m]
        return m

    @property
    def meridiansNum2LedNum(self)->Dict[int,List[int]]:
        m0:List[int]  = [154,145,133,121,112,108,109,115,127,139,151]
        m30:List[int] = [156,144,132,120,126,138,150,162]
        m60:List[int] = [155,143,131,119,111,114,125,137,149,161]
        m90:List[int] = [154,142,130,118,124,136,148,160]
        m120:List[int]= [159,147,135,123,113,110,117,129,141,153]
        m150:List[int]= [158,146,134,122,116,128,140,152]
        return {0:m0,30:m30,60:m60,90:m90,120:m120,150:m150}

    @property
    def ledStrList(self)->List[str]:
        return ['6-0','7-0','8-0','9-0','10-0','11-0','0-0','1-0','2-0','3-0','4-0','5-0',\
               '6-1','7-1','8-1','9-1','10-1','11-1','0-1','1-1','2-1','3-1','4-1','5-1',\
               '6-2','7-2','8-2','9-2','10-2','11-2','0-2','1-2','2-2','3-2','4-2','5-2',\
               '6-3','7-3','8-3','9-3','10-3','11-3','0-3','1-3','2-3','3-3','4-3','5-3',\
               '6-4','7-4','8-4','9-4','10-4','11-4','0-4','1-4','2-4','3-4','4-4','5-4',\
               '6-5','7-5','8-5','9-5','10-5','11-5','0-5','1-5','2-5','3-5','4-5','5-5',\
               '6-6','7-6','8-6','9-6','10-6','11-6','0-6','1-6','2-6','3-6','4-6','5-6',\
               '6-7','7-7','8-7','9-7','10-7','11-7','0-7','1-7','2-7','3-7','4-7','5-7',\
               '6-8','7-8','8-8','9-8','10-8','11-8','0-8','1-8','2-8','3-8','4-8','5-8']

    @property
    def meridianStr2ledNum(self)->Dict[str,int]:
        return {ledstr:i for i,ledstr in enumerate(self.ledStrList)}

    @property
    def initial_led_pattern_on(self)->List[bytearray]:
        leds:List[int] = [108,109,112]
        on:List[bytearray] = [self.generateByteArray(ledNum=ledNum,r=self.infrared_intensity,g=0,b=0) for ledNum in leds]
        return on

    @property
    def initial_led_pattern_off(self)->List[bytearray]:
        leds:List[int] = [108,109,112]
        off:List[bytearray] = [self.generateByteArray(ledNum=ledNum,r=0,g=0,b=0) for ledNum in leds]
        return off
    
    def infrareds(self)->Dict[str,bytearray]:
        on={i:self.generateByteArray(ledNum=i,r=self.infrared_intensity,g=0,b=0)  for i in range(108,163)}
        off={i:self.generateByteArray(ledNum=i,r=0,g=0,b=0)  for i in range(108,163)}
        return on,off

    def infrared_byteArrays(self,leds:List[int])->Dict[str,List[bytearray]]:
        if len(leds)>1:
            on:List[bytearray] = [self.infrareds[0][ledNum] for ledNum in leds]
            off:List[bytearray] = [self.infrareds[1][ledNum] for ledNum in leds]
        else:
            on:List[bytearray] = self.infrareds[0][leds[0]]
            off:List[bytearray] = self.infrareds[1][leds[0]]
        return {'on':on,'off':off}

    def flush(self)->None:
        leds:List[bytearray] = [bytearray]*163
        leds:List[bytearray] = [self.generateByteArray(ledNum=i,r=0,g=0,b=0) for i in range(163)]
        self.serial_port.writelines(leds)
        self.serial_port.write(self.update)

    def randomByteArrays(self)->List[Dict[str,List[bytearray]]]:
        return [self.generate_random_rgb_set() for i in range(1001)]

    def generate_random_rgb_set(self)->Dict[str,List[bytearray]]:
        i:int
        n:int
        subscenario:List[int] = [int]*12
        ledStrings:List[str] = [f'{i}-{random.randint(0,8)}' for i in range(12)]
        rgbrand:List[Tuple[int]] = [(random.randint(0,5),random.randint(0,3),random.randint(0,3)) for i in range(12)]
        rgbs:List[Tuple[int]] = [(35*mr,35*mg,35*mb) for mr,mg,mb in rgbrand]
        on:List[bytearray] = [self.generateByteArray(ledNum=self.meridianStr2ledNum[ledString],r=r,g=g,b=b) for ledString,(r,g,b) in zip(ledStrings,rgbs)]
        off:List[bytearray] = [self.generateByteArray(ledNum=self.meridianStr2ledNum[ledString],r=0,g=0,b=0) for ledString,(r,g,b) in zip(ledStrings,rgbs)]
        return {'on':on,'off':off}

