import cv2, numpy as np, tensorflow as tf
tf.config.optimizer.set_jit(enabled=True)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0],True)
tf.config.experimental.set_virtual_device_configuration(gpus[0],\
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])#1024

def load(json_path:str,weight_path:str)->None:
    json_file:_io.TextIOWrapper
    jsondata:str
    model:tf.keras
    e:float
    with open(json_path,'r') as json_file:
        jsondata = json_file.read()
        json_file.close()
    model = tf.keras.models.model_from_json(jsondata)
    model.load_weights(weight_path)
    return model



if __name__ == '__main__':
    json_path = '/home/ibex/Documents/refnet11_crop_5dioptri_meridian_16m_17.08.2021-10:09:31.json'
    h5_path = '/home/ibex/Documents/refnet11_crop_5dioptri_meridian_16m_17.08.2021-10:09:31.h5'

    model = load(json_path=json_path,weight_path=h5_path)
    model.save('/home/ibex/Documents/refnet11_crop_5dioptri_meridian_16m')