import serial
import cython
import time
from typing import *

@cython.cclass
class led_module():
    ser:serial.Serial
    illumination_command:bytes
    flush_command:bytes
    trigger_command:bytes
    trigger_state:int
    def __init__(self, 
                 animation_number:int=1,
                 animation_speed:int=14,
                 intensity:int=99,
                 trigger_delay:int=10,
                 led_delay:int=0) -> None:
        
        self.animation_speed = animation_speed
        self.intensity = intensity
        self.animation_number = animation_number
        self.trigger_delay = trigger_delay
        self.led_delay = led_delay
        
        self.ser = self.open()
        self.illumCmd()
        self.flushCmd()
        self.trgCmd()

    def open(self) -> serial.Serial:
        # Configure the serial port settings
        ser:serial.Serial = serial.Serial(
            port='/dev/ttyTHS1',  
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=None)
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
                             int(hex(self.animation_number),16)]
        
        command.append(self.calculate_checksum(command))
        self.trigger_command:bytes = bytes(command)

    def illuminate(self) -> None:
        # turns three central leds on. trigger is sent one on every call.
        self.ser.write(self.illumination_command)   

    def flush(self, trigger_state:int=1)->None:
        # turns all lwds off. 
        # if trigger_state==1 card sends a trigger to camera while turning all leds off, 
        # else it turns all leds off without sending a trigger to camera
        # print('------0')
        if trigger_state!=self.trigger_state:
            self.flush_command(trigger_state)
        # print('------1')
        self.ser.write(self.flush_command)
        # print('------2')
        
    def trigger(self)->None:        
        time.sleep(0.05)
        self.ser.write(self.trigger_command)

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



if __name__=='__main__':
    animation_number=1
    animation_speed=15
    intensity=99
    trigger_delay=3
    led_delay=1
    ledCard = led_module(animation_number=animation_number,
                         animation_speed=animation_speed,
                         intensity=intensity,
                         trigger_delay=trigger_delay,
                         led_delay=led_delay)
    try:
        while True:
            prompt = input('which function do you want to test? \n \n --flush :      0 \n --illuminate : 1  \n --trigger :    2 \n --single_trigger : 3 \n --stop :       e \n :/>')
            if prompt=='e':
                ledCard.ser.close()
                break
            if int(prompt)==1:
                ledCard.illuminate()
            elif int(prompt)==0:
                ledCard.flush()
            elif int(prompt)==2:
                ledCard.trigger()
            elif int(prompt)==3:
                while True:
                    lednum = input('Enter led Number [0-55]:/> [int or "e"]')
                    if lednum=='e':
                        
                        break
                    ledCard.single_trigger(int(lednum))
            
    except KeyboardInterrupt:
        ledCard.ser.close()
        print('!!FINISH!!')