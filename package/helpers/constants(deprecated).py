from typing import Any
import cv2
from ..configs import read_confs
confs = read_confs.configs_()

class Constants:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    
    def __setattr__(self, name, value):
        if hasattr(self, name):
            raise AttributeError("Cannot modify constant value.")
        else:
            super().__setattr__(name, value)
    

width = int(confs.camera_conf['width'] * confs.device_conf['scale_percent'] / 100)
height = int(confs.camera_conf['width'] * confs.device_conf['scale_percent'] / 100)

# Create an instance of the Constants class
constants = Constants(scale_percent=confs.camera_conf['scale_percent'],
                      acquisition_number=confs.device_conf['acquisition_number'],
                      width = int(724 * confs.scale_percent / 100),
                      height = int(1280 * confs.scale_percent / 100),
                      cv2Version = cv2.__version__[0],
                      pixel2mm = 0.0048,
                      distance = 1000,
                      focal_length = 39.2,
                      scale = (confs.device_conf['distance']-confs.camera_conf['focal_length'])/confs.camera_conf['focal_length']
)

