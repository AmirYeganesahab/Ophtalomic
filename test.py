# test_my_module.py
# import sys
# sys.path.append('/home/ibex/Documents/OphtalomogicDevice/package')
import unittest

from main import (device_conf,
                    ledboard_conf,
                    camera_conf,
                    width,
                    height)

class TestAddFunction(unittest.TestCase):

    def test_device_configs(self):
        #just print ot ber sure the confrs are loaded successfully. 
        #if any None occures it means a problem
        self.assertIsNotNone(device_conf)

    def test_ledboard_configs(self):
        self.assertIsNotNone(ledboard_conf)
        
    def test_camera_configs(self):
        self.assertIsNotNone(camera_conf)
    
    def test_frame_width(self):
        self.assertIsNotNone(width)
    
    def test_frame_height(self):
        self.assertIsNotNone(height)

    
if __name__ == '__main__':
    unittest.main()