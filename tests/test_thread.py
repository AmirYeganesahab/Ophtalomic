from tensorrt_code import *
import threading
class myThread(threading.Thread):
    args:list
    func:TRTInference.infer
    def __init__(self, func, args)->None:
        threading.Thread.__init__(self)
        self.func = func
        self.args = args
    def run(self)->None:
    #   print ("Starting " + self.args[0])
        self.func(*self.args)

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