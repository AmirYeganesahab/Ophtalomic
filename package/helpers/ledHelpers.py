from typing import *
import numpy as np

def calculate_checksum(data_bytes):
    # Calculate the XOR checksum of the data bytes
    checksum = 0
    for byte in data_bytes:
        checksum ^= byte
    return checksum

def generate_led_list():
        # Generates list of leds with custom arrangement
        M0 = [0,1,2,3,4,5,6,7,8,9,10][::-1]
        M30 = [11,12,13,14,14,5,15,15,16,17,18][::-1]
        M60 = [19,20,21,22,23,5,24,25,26,27,28][::-1]
        M90 = [29,30,31,31,32,5,33,33,34,35,36][::-1]
        M120 = [37,38,39,40,41,5,42,43,44,45,46]
        M150 = [47,48,49,50,50,5,51,51,52,53,54]
        return [*M0,*M30,*M60,*M90,*M120,*M150] # in case you want to run on old led card return [*M0,*M60,*M120]

def get_fois(frames):
    lois = generate_led_list() # leds of interest
    # gets frames of interest based on leds of interest(lois)
    output:List[np.ndarray]=[np.ndarray]*len(lois)
    for i,led in enumerate(lois):
        output[i]=frames[led]
    return output