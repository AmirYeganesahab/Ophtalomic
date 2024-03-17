from __future__ import print_function
from inspect import currentframe

from ximea import xiapi
import cython, logging,os
from typing import *
import serial
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logging.info('__ sensors file called __')
import random
import numpy as np
import time

def get_linenumber()->str:
    cf:currentframe = currentframe()
    return 'line: '+str(cf.f_back.f_lineno)

def write_log(ss)->None:
    with open("/home/ibex/Documents/XCamera_modern.v2.0/log.txt", "a") as fp:
        fp.write(str(ss)+"\n")

print('')

@cython.cclass
class camera_():
    logger:logging.Logger
    trigger_source:str
    exposure_time:int
    trigger:Dict[str,str]
    gpi:Dict[str,str]
    gpo:Dict[str,str]
    general:Dict[str,Union[int,float]]
    ioi:List[int]
    led_module:led_module
    device:xiapi.Camera
    img: xiapi.Image
    ledboard_timeout:float
    def __init__(self,
                exposure_time:cython.int = int(15000),
                trigger_source:str='hardware',
                animation_number:int=1,
                animation_speed:int=14,
                intensity:int=99,
                trigger_delay:int=10,
                led_delay:int=0,
                ledboard_timeout:float=0.05) -> None:
        self.generate_led_list()
        self.logger = logging.getLogger('camera logs')
        self.trigger_source = trigger_source
        self.exposure_time = exposure_time
        
        self.open_device()

        self.led_module = led_module(animation_number=animation_number,
                                        animation_speed=animation_speed,
                                        intensity=intensity,
                                        trigger_delay=trigger_delay,
                                        led_delay=led_delay,
                                        timeout=ledboard_timeout)

    def open_device(self)->None:
        # print(get_linenumber())
        self.logger.info(f'trigger source: {self.trigger_source}')
        # print(get_linenumber())
        self.device = xiapi.Camera()
        # print(get_linenumber())
        self.settings()
        # print(get_linenumber())
        try:
            self.device.close_device()
        except:
            pass
        # print(get_linenumber())
        self.logger.info('device is openning ...')
        # print(get_linenumber())
        self.device.open_device()
        # print(get_linenumber())
        self.logger.info('device is up')
        # print(get_linenumber())
        self.device.stop_acquisition()
        # print(get_linenumber())
        self.apply_settings()
        # print(get_linenumber())
        self.img = xiapi.Image()
        # print(get_linenumber())
        return self.device,self.img

    def apply_settings(self)->None:
        # trigger_source
        self.device.set_param( 'trigger_source',self.trigger['source'])
        self.logger.info(f'trigger_source set to: {self.trigger["source"]}')
        # Trigger Selector
        self.device.set_param('trigger_selector',self.trigger['selector'])
        self.logger.info(f'trigger_selector set to: {self.trigger["selector"]}') 
        # GPI Mode
        self.device.set_param( 'gpi_mode',self.gpi['mode'])
        self.logger.info(f'gpi_mode set to: {self.gpi["mode"]}') 
        # GPI Selector
        self.device.set_param( 'gpi_selector',self.gpi['selector'])
        self.logger.info(f'gpi_selector set to: {self.gpi["selector"]}')
        # GPO Mode
        self.device.set_param( 'gpo_mode',self.gpo['mode'])
        self.logger.info(f'gpo_mode set to: {self.gpo["mode"]}') 
        # Auto Exposure
        self.device.set_param('aeag',self.general['auto_exposure'])# or 1 AutoExposure
        self.logger.info(f'Auto exposure set to: {"on" if self.general["auto_exposure"] else "off"}')
        # Exposure Time
        if not self.general['auto_exposure']:
            self.device.set_param('exposure',self.general['exposure_time'])
            self.logger.info(f'exposure_time set to: {self.general["exposure_time"]}')
        else:
            # Maximum limit of gain in AEAG procedure.
            self.device.set_param( 'ag_max_limit',self.general['ag_max_limit'])#db
            self.logger.info(f'ag_max_limit set to: {self.general["ag_max_limit"]}')
            # Maximum limit of exposure (in uSec) in AEAG procedure.
            self.device.set_param( 'ae_max_limit',self.general['ae_max_limit'])#us
            self.logger.info(f'ae_max_limit set to: {self.general["ae_max_limit"]}')
            # Exposure priority for Auto Exposure / Auto Gain function.
            self.device.set_param( 'exp_priority',self.general['exp_priority'])# maximum 1
            self.logger.info(f'exp_priority set to: {self.general["exp_priority"]}')
            # Average intensity of output signal AEAG should achieve(in %).
            self.device.set_param( 'aeag_level',self.general['aeag_level'])
            self.logger.info(f'aeag_level set to: {self.general["aeag_level"]}')
        # Description: Activates Look-Up-Table (LUT).
        # Note1: Possible value: 0 - sensor pixels are transferred directly
        # Note2: Possible value: 1 - sensor pixels are mapped through LUT
        self.device.set_param( 'LUTEnable',self.general['LUTEnable'])
        self.logger.info(f'LUTEnable set to: {self.general["LUTEnable"]}')
        
        #self.device.set_param('acq_frame_burst_count',1)
        #self.device.set_exposure_burst_count(self.general['exposure_burst_count'])

    def settings(self)->None:
        self.trigger = {
                        'source':"XI_TRG_SOFTWARE" if self.trigger_source=='software'\
                                                    else "XI_TRG_EDGE_RISING",
                        'selector':"XI_TRG_SEL_FRAME_START",
                        }
        self.gpi = {'selector':"XI_GPI_PORT1",
                    'mode':"XI_GPI_TRIGGER"
                    }
        
        self.gpo = {'lsector':"XI_GPO_PORT1",
                    'mode':"XI_GPO_EXPOSURE_ACTIVE_NEG",
                    'mode_':"XI_GPO_BUSY_NEG"
                    }
        # ag_max_limit: Maximum limit of gain in AEAG procedure.(db)
        # ae_max_limit: Maximum limit of exposure (in uSec) in AEAG procedure.
        
        self.general = {'auto_exposure':0,
                        'exposure_time':self.exposure_time,
                        'ag_max_limit':0,
                        'ae_max_limit':15000,
                        'exp_priority':0.5,
                        'aeag_level':50,
                        'LUTEnable':0,
                        'exposure_burst_count':1,
                        }
        
    def get_image(self)->np.ndarray:
        #self.device.start_acquisition()
        # self.device.set_param('trigger_software',1)
        self.device.get_image(self.img)
        return self.img.get_image_data_numpy()

    def generate_led_list(self):
        M0 = [1,2,3,4,5,6,7,8,9,10,11][::-1]
        M60 = [20,21,22,23,24,25,26,27,28,29][::-1]
        M120 = [38,39,40,41,42,43,44,45,46,47]
        # M0 = [4,5,6,7,8,9,10,11,12,13,14][::-1]
        # M60 = [23,24,25,26,27,28,29,30,31,32][::-1]
        # M120 = [41,42,43,44,45,46,47,48,49,50]
        # #ioi = [0,1,2,3,4,5,6,7,8,9,10,19,20,21,22,23,24,25,26,27,28,37,38,39,40,41,42,43,44,45,46]
        # #ioi = [10,9,8,7,6,5,4,3,2,1,0,28,27,26,25,24,23,22,21,20,19,46,45,44,43,42,41,40,39,38,37]
        self.ioi=[*M0,*M60,*M120]
        
    def capture_gen(self)->np.ndarray:
        for i in range(55):
            yield self.get_image()#,self.led_module.ser.read(8)

    def capture_single(self,led_number:int)->np.ndarray:
        self.led_module.single_trigger(led_number)
        return self.get_image()

    def capture_generator(self)->np.ndarray:
        self.device.stop_acquisition()
        self.led_module.flush()
        self.device.start_acquisition()
        leds:List[int]=[int]*55
        led:int
        # imgs:List[np.ndarray] = [np.zeros((1024, 1280))]*55
        # img = np.zeros((1024,1280))
        for led in self.ioi:
            self.led_module.single_trigger(led)

            yield self.get_image()

                
    def illuminate(self)->np.ndarray:
        # this module is used to turn 3 central leds on and capture image
        # befor calling this you may need to start acquisition
        if self.device.get_acquisition_status()=='XI_OFF':
                self.device.start_acquisition()
        ret= self.led_module.illuminate()
        return ret,self.get_image()
    
    def close_device(self)->None:
        self.device.stop_acquisition()
        self.device.close_device()


@cython.cclass
class led_module():
    ser:serial.Serial
    illumination_command:bytes
    flush_command:bytes
    trigger_command:bytes
    trigger_state:int
    animation_speed:int
    intensity:int
    animation_number:int
    trigger_delay:int
    led_delay:int
    set_daley_command:bytes
    timeout:float
    def __init__(self, 
                 animation_number:int=1,
                 animation_speed:int=14,
                 intensity:int=99,
                 trigger_delay:int=10,
                 led_delay:int=5,
                 timeout:float=0.05) -> None:
        
        self.animation_speed = animation_speed
        self.intensity = intensity
        self.animation_number = animation_number
        self.trigger_delay = trigger_delay
        self.led_delay = led_delay
        self.timeout = timeout
        
        self.ser = self.open()
        self.setDelayCmd()
        self.illumCmd()
        self.flushCmd()
        self.trgCmd()
        self.set_daleys()

    def open(self) -> serial.Serial:
        # Configure the serial port settings
        ser:serial.Serial = serial.Serial(
            port='/dev/ttyTHS1',  
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.timeout)
        return ser
        
    def calculate_checksum(self,data_bytes:List[int]) -> int:
        # Calculate the XOR checksum of the data bytes
        byte:int
        checksum:int = 0
        for byte in data_bytes:
            checksum ^= byte
        return checksum

    def illumCmd(self)->None:
        # Aydınlatma modu
        # this module is used to turn 3 central leds on.
        # protokol= {'command to be sent':0xaa,
        #             'run auto burst':0x01,
        #             'N/A':0x00,
        #             'N/A':0x00,
        #             'led intensity':int(hex(100),16),
        #             'animation speed':int(hex(14),16),
        #             'animation number':0x01,
        #             'checksum':0x00}
        command:List[int] = [0xaa,
                             0x01,
                             0x00,
                             0x00,
                             int(hex(self.intensity), 16),
                             int(hex(self.animation_speed),16),
                             int(hex(self.animation_number),16)]
        
        command.append(self.calculate_checksum(command))
        self.illumination_command = bytes(command)
        
    def flushCmd(self,trigger_state:int=1)->None:
        # trigger_state is a binary value with 1 as active trigger
        # all leds off example (animation is on)
        # protokol= {'command to be sent':0xaa,
        #             'turn all leds off':0x05,
        #             'N/A':0x00,
        #             'N/A':0x00,
        #             'trigger on/off':int(hex(trigstate),16),
        #             'animation speed':int(hex(20),16),
        #             'animation number':0x01,
        #             'checksum':0x00}
        self.trigger_state=trigger_state
        all_off:List[int]= [0xaa,
                            0x05,
                            0x00,
                            0x00,
                            int(hex(trigger_state),16),
                            int(hex(self.animation_speed),16),
                            int(hex(self.animation_number),16)]
        
        all_off.append(self.calculate_checksum(all_off))
        self.flush_command = bytes(all_off)

    def trgCmd(self)->None:

        command:List[int] = [0xaa,
                             0x02,
                             0x00,
                             0x01,
                             int(hex(self.intensity), 16),
                             int(hex(self.animation_speed),16),
                             int(hex(0),16)]
        
        command.append(self.calculate_checksum(command))
        self.trigger_command:bytes = bytes(command)

    def setDelayCmd(self)->None:
        command:List[int] = [0xaa,
                             0x03,
                             0x00,
                             int(hex(self.trigger_delay),16),
                             int(hex(self.led_delay),16),
                             0x00,
                             0x00]

        command.append(self.calculate_checksum(command))
        self.set_daley_command = bytes(command)

    def illuminate(self) -> None:
        # turns three central leds on. trigger is sent one on every call.
        self.ser.write(self.illumination_command)
        return self.ser.read(8)

    def flush(self, trigger_state:int=1)->None:
        # turns all lwds off. 
        # if trigger_state==1 card sends a trigger to camera while turning all leds off, 
        # else it turns all leds off without sending a trigger to camera
        # print('------0')
        if trigger_state!=self.trigger_state:
            self.flushCmd(trigger_state)
        # print('------1')
        self.ser.write(self.flush_command)
        return self.ser.read(8) 
        # print('------2')
        
    def trigger(self)->None:
        self.ser.write(self.trigger_command)
         
    def set_daleys(self)->None:
        time.sleep(0.05)
        self.ser.write(self.set_daley_command)
        time.sleep(0.05)

    def single_trigger(self,led_number:int)->None:
        # 0xAA	0x06	0x00	TrigDelay	LedDelay	Parlaklık	IR Led No	Byte1 xor Byte1 xor … xor Byte7
        # protokol = {'command to be sent':0xaa,
        #             'run auto burst':0x06,
        #             'N/A':0x00,
        #             'TrigDelay':int(hex(TrigDelay),16),
        #             'LedDelay':int(hex(LedDelay),16),
        #             'Parlaklık':int(hex(Parlaklık),16),
        #             'IR Led No':ledNum,
        #             'checksum':0x00}
        command = [0xaa,
                   0x06,
                   0x00,
                   int(hex(self.trigger_delay),16),
                   int(hex(self.led_delay),16),
                   int(hex(self.intensity),16),
                   int(hex(led_number),16)]
        command.append(self.calculate_checksum(command))
        command_bytes = bytes(command)
        # print(command_bytes)
        self.ser.write(command_bytes)
        time.sleep(0.1)

