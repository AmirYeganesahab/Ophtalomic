from PIL import Image
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import threading
import time
import math
import cv2
import cython
import _io
from typing import *

exitFlag = 0
class TRTInference:
    cfx:pycuda._driver.Context
    stream:pycuda._driver.Stream
    context:trt.tensorrt.IExecutionContext
    engine:trt.tensorrt.ICudaEngine
    host_inputs:List[np.ndarray]
    cuda_inputs:List[pycuda._driver.DeviceAllocation]
    host_outputs:List[np.ndarray]
    cuda_outputs:List[pycuda._driver.DeviceAllocation]
    bindings:List[int]
    output:np.ndarray

    def __cinit__(self, trt_engine_path:str, trt_engine_datatype:trt.tensorrt.DataType, batch_size:int)->None:
        
        f:_io.BufferedReader
        binding:str
        self.cfx = cuda.Device(0).make_context()
        stream:pycuda._driver.Stream = cuda.Stream()
        TRT_LOGGER:trt.tensorrt.Logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        runtime:trt.tensorrt.Runtime = trt.Runtime(TRT_LOGGER)
        # deserialize engine
        with open(trt_engine_path, 'rb') as f:
            buf:bytes = f.read()
            engine:trt.tensorrt.ICudaEngine = runtime.deserialize_cuda_engine(buf)
        context:trt.tensorrt.IExecutionContext = engine.create_execution_context()
        
        # prepare buffer
        host_inputs:List[np.ndarray]= []
        cuda_inputs:List[pycuda._driver.DeviceAllocation]  = []
        host_outputs:List[np.ndarray] = []
        cuda_outputs:List[pycuda._driver.DeviceAllocation] = []
        bindings:List[int] = []

        for binding in engine:
            size:int = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            host_mem:np.ndarray = cuda.pagelocked_empty(size, np.float32)
            cuda_mem:pycuda._driver.DeviceAllocation = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(cuda_mem))
            if engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # store
        self.stream  = stream
        self.context = context
        self.engine  = engine

        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings


    def infer(self, input_img_path:str)->None:
        threading.Thread.__init__(self)
        self.cfx.push()

        # restore
        stream:pycuda._driver.Stream  = self.stream
        context:trt.tensorrt.IExecutionContext = self.context
        engine:trt.tensorrt.ICudaEngine = self.engine

        host_inputs:List[np.ndarray] = self.host_inputs
        cuda_inputs:List[pycuda._driver.DeviceAllocation] = self.cuda_inputs
        host_outputs:List[np.ndarray] = self.host_outputs
        cuda_outputs:List[pycuda._driver.DeviceAllocation] = self.cuda_outputs
        bindings:List[int] = self.bindings

        # read image
        image:np.ndarray = roi_preprocess(cv2.imread(input_img_path,0)).roiNet_input

        np.copyto(host_inputs[0], image.ravel())

        # inference
        # start_time = time.time()
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        stream.synchronize()
        # print("execute times "+str(time.time()-start_time))
        if len(host_outputs[0])<10:
            self.output = host_outputs[0]
        else:
            # parse output
            self.output =  host_outputs[0].reshape((128, 256,1))
        
        self.cfx.pop()
        
    def destory(self)->None:
        self.cfx.pop()




@cython.cclass
class roi_preprocess():
    roiNet_input:np.ndarray
    def __cinit__(self,image:np.ndarray)->None:
        self.roiNet_input = self.resize(image) 

    def resize(self,image:np.ndarray)->np.ndarray:
        image = cv2.resize(image,(256,128),interpolation=cv2.INTER_AREA) #width,height 
        image = np.expand_dims(image,0).astype(np.float32)
        image = np.expand_dims(image,-1)
        return image/255

if __name__ == '__main__':
    # Create new threads
    '''
    format thread:
        - func: function names, function that we wished to use
        - arguments: arguments that will be used for the func's arguments
    '''

    trt_engine_path = '/home/ibex/Documents/trt_models/roinet_256x128_annotationsiz2.trt'

    max_batch_size = 1
    trt_inference_wrapper = TRTInference(trt_engine_path,
        trt_engine_datatype=trt.DataType.FLOAT,
        batch_size=max_batch_size)

    # Get TensorRT SSD model output
    input_img_path = '/home/ibex/Documents/dummy_data/N852_(4, 5, 6).png'
    thread1 = myThread(trt_inference_wrapper.infer, [input_img_path])

    # Start new Threads
    thread1.start()
    thread1.join()
    print(trt_inference_wrapper.output.shape)
    trt_inference_wrapper.destory()
    print ("Exiting Main Thread")