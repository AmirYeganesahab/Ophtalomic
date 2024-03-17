import pycuda.driver as cuda
import pycuda.autoinit
import imutils,traceback
from typing import *
import tensorrt as trt
trt.Logger(trt.ILogger.ERROR)
import os,sys,threading,cv2, random,cython, logging,_io,numpy as np, time,asyncio
import matplotlib.pyplot as plt

pixel2mm:cython.float = 0.0048
distance:cython.float = 1000
focal_length:cython.float = 39.2
scale:cython.float = (distance-focal_length)/focal_length
cv2version = int(cv2.__version__[0])


from inspect import currentframe
def get_linenumber()->str:
    cf = currentframe()
    return 'line: '+str(cf.f_back.f_lineno)

def write_log(ss)->None:
    with open("log.txt", "a") as fp:
        fp.write(str(ss)+"\n")

@cython.cclass
class load():
    engine: trt.tensorrt.ICudaEngine 
    def __init__(self, plan_path:str)->None:
        f:_io.BufferedReader
        engine_data: bytes
        with open(plan_path, 'rb') as f:
            engine_data = f.read()  
            f.close() 
        TRT_LOGGER:trt.tensorrt.Logger = trt.Logger(trt.Logger.WARNING)
        trt_runtime:trt.tensorrt.Runtime = trt.Runtime(TRT_LOGGER)
        self.engine = trt_runtime.deserialize_cuda_engine(engine_data)

@cython.cclass
class buffer():
    """
    The TensorRT engine runs inference in the following workflow: 
    Allocate buffers for inputs and outputs in the GPU.
    Copy data from the host to the allocated input buffers in the GPU.
    Run inference in the GPU. 
    Copy results from the GPU to the host. 
    Reshape the results as necessary. 
    """
    engine:trt.tensorrt.ICudaEngine
    data_type:trt.tensorrt.DataType
    batch_size:int
    h_input_1:np.ndarray
    h_output:np.ndarray
    d_input_1:pycuda._driver.DeviceAllocation
    d_output:pycuda._driver.DeviceAllocation

    def __init__(self, engine:trt.tensorrt.ICudaEngine,\
                    batch_size:int,\
                    dtype:trt.tensorrt.DataType)->None:
        self.engine = engine
        self.data_type = dtype
        self.batch_size = batch_size 
        self.allocate_buffers(batch_size)

    def allocate_buffers(self, batch_size:int)->None:

        """
        This is the function to allocate buffers for input and output in the device
        Args:
            engine : The path to the TensorRT engine. 
            batch_size : The batch size for execution time.
            data_type: The type of the data for input and output, for example trt.float32. 

        Output:
            h_input_1: Input in the host.
            d_input_1: Input in the device. 
            h_output_1: Output in the host. 
            d_output_1: Output in the device. 
            stream: CUDA stream.
        """

        # Determine dimensions and create page-locked memory buffers (which won't be swapped to disk) to hold host inputs/outputs.
        #input/output shape belirleme - page-locked memory buffers oluşturur
        self.h_input_1 = cuda.pagelocked_empty(batch_size * trt.volume(self.engine.get_binding_shape(0)),\
                                        dtype=trt.nptype(self.data_type))
        self.h_output = cuda.pagelocked_empty(batch_size * trt.volume(self.engine.get_binding_shape(1)),\
                                        dtype=trt.nptype(self.data_type))
        # Allocate device memory for inputs and outputs (host).
        self.d_input_1 = cuda.mem_alloc(self.h_input_1.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        # Create a stream in which to copy inputs/outputs and run inference.
        

    def load_images_to_buffer(self,pics:np.ndarray, pagelocked_buffer:np.ndarray)->None:
        #görüntüleri buffere yazar
        #print("inference laod images to buffer before shape : ",np.shape(pics))
        pics:np.ndarray = np.asarray(pics).ravel()
        #print("inference laod images to buffer after shape : ",np.shape(pics))
        np.copyto(pagelocked_buffer, pics) #des,source (pre'deki değerleri pagel atıyor. pagel değişiyor.)

    def do_inference(self,pics_1:np.ndarray, height:int=0, width:int=0)->np.ndarray:
        context:trt.tensorrt.IExecutionContext
        stream:pycuda._driver.Stream
        """
        This is the function to run the inference
        Args:
            engine : Path to the TensorRT engine 
            pics_1 : Input images to the model.  
            h_input_1: Input in the host         
            d_input_1: Input in the device 
            h_output_1: Output in the host 
            d_output_1: Output in the device 
            stream: CUDA stream
            batch_size : Batch size for execution time
            height: Height of the output image
            width: Width of the output image

        Output:
            The list of output images
        """
        stream = cuda.Stream()
        self.load_images_to_buffer(pics_1, self.h_input_1)
        with self.engine.create_execution_context() as context:
            
            # Transfer input data to the GPU.  (input images gpuya transferi)
            cuda.memcpy_htod_async(self.d_input_1, self.h_input_1, stream)
            # Run inference.
            context.profiler = trt.Profiler()
            # for inference
            context.execute(batch_size=1, bindings=[int(self.d_input_1), int(self.d_output)])
            # Transfer predictions back from the GPU. (output gpu'da hosta transfer bu sefer)
            cuda.memcpy_dtoh_async(self.h_output, self.d_output, stream)
            # Synchronize the stream
            stream.synchronize()
            # Return the host output.
            if height==0:
                return self.h_output
                
            return self.h_output.reshape((height, width,self.batch_size))

@cython.cclass
class roiNet():
    engine:trt.tensorrt.ICudaEngine
    buffer:buffer
    H:int
    W:int
    n_random_crop:int
    
    def __init__(self,path:str,roiH:int=128,roiW:int=256, n_random_crop:int=1)->None:
        self.engine = load(path).engine
        self.H, self.W = roiH,roiW
        self.n_random_crop = n_random_crop
        self.buffer = buffer(self.engine, 1, trt.float32)

    def inference(self,Oframe:np.ndarray)->Tuple[Tuple[Tuple[float,float],Tuple[float,float]],Tuple[np.ndarray,np.ndarray]]:
        #roi:np.ndarray
        hw:List[int] = Oframe.shape[:2]
        # t:float = time.time()
        # write_log(f'nn:{get_linenumber()}')
        frame:np.ndarray= roi_preprocess(Oframe).roiNet_input
        # write_log(f'nn:{get_linenumber()}')
        Roi_image:np.ndarray =  self.buffer.do_inference(frame, self.H, self.W)
        # write_log(f'nn:{get_linenumber()}')
        postprocess:roi_postprocess = roi_postprocess(frame=Oframe,
                                  inp = Roi_image,
                                  nrc = self.n_random_crop,
                                  shp=hw)
        # write_log(f'nn:{get_linenumber()}')
        try:
            # write_log(f'nn:{get_linenumber()}')
            return postprocess.locate_eyes()
            
        except Exception as e:
            write_log(f'nn:{get_linenumber()}')
            write_log(str(e))
            return (),()

@cython.cclass
class roi_preprocess():
    roiNet_input:np.ndarray
    def __init__(self,image:np.ndarray)->None:
        self.roiNet_input = self.resize(image) 

    def resize(self,image:np.ndarray)->np.ndarray:
        image:np.ndarray = cv2.resize(image,(256,128),interpolation=cv2.INTER_AREA) #width,height 
        image = np.expand_dims(image,0).astype(np.float32)
        image = np.expand_dims(image,-1)
        return image/255

@cython.cclass
class roi_postprocess():
    input:np.ndarray
    thr:Tuple[float,float]
    n_random_crops:int
    shp:List[int]
    frame:np.ndarray
    input:np.ndarray
    thr:Tuple[float]
    def __init__(self,
                    frame:np.ndarray,
                    inp:np.ndarray,
                    nrc:int,
                    shp:Tuple[int,int],
                    default_threshold:Tuple[float,float]=(.3,1.))->None:

        self.n_random_crops = nrc
        self.shp = shp
        self.frame = frame
        self.input = inp
        self.thr = default_threshold

    def pupil_is_valid(self,
                        p:Tuple[float,float],
                        inlyer:int=150)->bool:
        #print(p, inlyer)
        if all(np.array(p)>inlyer) and \
                p[0]<1280-inlyer and \
                p[1]<1024-inlyer and \
                p[0] !=0 and p[1] !=0:
            return True
        return False
        
    def correct_outlier_roi(self, 
                            pupil:Union[Tuple[float,float],List[float]],
                            shp:Tuple[int,int],
                            halphsize:int=128)->Tuple[float,float]:
        pupil = list(pupil)
        if pupil[0]<halphsize: 
            pupil[0] = halphsize
        if pupil[0]+128>=shp[1]:
            pupil[0] = shp[1]-halphsize
        if pupil[1]<halphsize: 
            pupil[1] = halphsize
        if pupil[1]+halphsize>=shp[0]:
            pupil[1] = shp[0]-halphsize
        return tuple(pupil)
            
    def angle_is_acceptable(self,
                            left:Tuple[float,float],
                            right:Tuple[float,float], 
                            threshold:int = 20)->bool:
        d:np.ndarray
        pdist:float
        alpha:float

        #d = (right[0]-left[0],right[1]-left[1])
        d = np.subtract(left,right)
        #print(d)
        pdist = np.linalg.norm(d)*pixel2mm*scale #mm
        #print(pdist)
        if 30 > pdist > 100:
            return False
        alpha = np.rad2deg(np.arctan(d[1]/d[0]))
        #print('lapha',alpha)
        if abs(alpha) > threshold:
            return False
        return True
    
    def locate_eyes(self)->Tuple[Tuple[Tuple[float,float],Tuple[float,float]],Tuple[np.ndarray,np.ndarray]]:
        binary:np.ndarray
        c:np.ndarray
        ellipse0:Tuple[Tuple[float,float],Tuple[float,float],float]
        ellipse1:Tuple[Tuple[float,float],Tuple[float,float],float]
        # write_log(f'nn:{get_linenumber()}')
        if self.n_random_crops!=1:
            halphsize = 200
        else:
            halphsize = 128
        # write_log(f'nn:{get_linenumber()}')
        _,binary = cv2.threshold(self.input,
                                    self.thr[0],
                                    self.thr[1], 
                                    cv2.THRESH_BINARY)
        # write_log(f'nn:{get_linenumber()}')
        
        if cv2version not in [2,4]:
            _,c,_ = cv2.findContours(binary.astype(np.uint8),
                                    cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)
            # write_log(f'nn:{get_linenumber()}')
        else:
            c,_ = cv2.findContours(binary.astype(np.uint8),
                                    cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)
            # write_log(f'nn:{get_linenumber()}')
        areas:List[float] = [cv2.contourArea(a) for a in c]
        # write_log(f'nn:{get_linenumber()}')
        # print('predict_rois233')
        s:List[float] = np.sort(areas)
        if len(s)<1: 
            return (),()
        # write_log(f'nn:{get_linenumber()}')
        scale:np.ndarray = np.divide(self.shp,
                                    self.input.shape[:2])[::-1]
        # write_log(f'nn:{get_linenumber()}')
        ellipse0 = cv2.fitEllipse(np.squeeze(c[areas.index(s[-1])],1))
        # write_log(f'nn:{get_linenumber()}')
        p0:Union[np.ndarray, Tuple[float,float]] = np.multiply(ellipse0[0],scale)
        # write_log(f'nn:{get_linenumber()}')
        p0 = tuple([int(round(p0[0])), int(round(p0[1]))])
        # write_log(f'nn:{get_linenumber()}')
        p0 = self.correct_outlier_roi(p0,self.shp,halphsize = halphsize)
        # write_log(f'nn:{get_linenumber()}')
        # logging.info(s)
        if len(s)>=2:
            ellipse1 = cv2.fitEllipse(np.squeeze(c[areas.index(s[-2])],1))
            # write_log(f'nn:{get_linenumber()}')
            # print('predict_rois233')
            p1:Union[np.ndarray, Tuple[float,float]] = np.multiply(ellipse1[0],scale)
            # write_log(f'nn:{get_linenumber()}')
            p1 = tuple([int(round(p1[0])), int(round(p1[1]))])
            # write_log(f'nn:{get_linenumber()}')
            p1 = self.correct_outlier_roi(p1,self.shp,halphsize = halphsize)
            # write_log(f'nn:{get_linenumber()}')
            # logging.info(f'p0:{p0},p1:{p1}')
            # logging.info(f'self.pupil_is_valid(p0):{self.pupil_is_valid(p0)}')
            # logging.info(f'self.pupil_is_valid(p1):{self.pupil_is_valid(p1)}')
            #print('self.pupil_is_valid(p0)',self.pupil_is_valid(p0))
            #print('self.pupil_is_valid(p1)',self.pupil_is_valid(p1))
            if self.pupil_is_valid(p0) and self.pupil_is_valid(p1):
                # write_log(f'nn:{get_linenumber()}')
                if p0[0]<p1[0]:
                    right_center,left_center = p0,p1
                else:
                    right_center,left_center = p1,p0
                # write_log(f'nn:{get_linenumber()}')
                # logging.info(f'self.angle_is_acceptable(left_center,right_center):{self.angle_is_acceptable(left_center,right_center)}')
                #print(left_center,right_center)
                if not self.angle_is_acceptable(left_center,right_center):
                    return (),()   
                # write_log(f'nn:{get_linenumber()}')
            # write_log(f'nn:{get_linenumber()}')
        elif len(s)==1 and self.pupil_is_valid(p0):
            # write_log(f'nn:{get_linenumber()}')
            if p0[0]<self.input.shape[0]/2:
                right_center,left_center = p0, ()
            else:
                right_center,left_center = (), p0
            # write_log(f'nn:{get_linenumber()}')
        else: 
            # write_log(f'nn:{get_linenumber()}')
            return (),()
        # write_log(f'nn:{get_linenumber()}')

        return left_center, right_center

@cython.cclass
class pupilNet():
    engine:trt.tensorrt.ICudaEngine
    halphsize:int
    shp:Tuple[float,float]
    H:int
    W:int
    buffer:buffer
    
    def __init__(self,path:str,pH:int=256,pW:int=256,halphsize:int=128)->None:
        self.engine = load(path).engine
        self.halphsize=halphsize
        self.H, self.W = pH,pW
        self.buffer = buffer(self.engine, 1, trt.float32)
    
    def inference(self,
                Oframe:np.ndarray,
                centers:Tuple[Tuple[float,float],Tuple[float,float]])->Tuple[Dict[str,np.ndarray],\
                                                                        Dict[str,\
                                                                        Union[Tuple[float], float, float, Dict[str,float]]]]:
        preprocess:pupil_preprocess
        postprocess:Dict[str,Union[Tuple[Tuple[float,float],Tuple[float,float],float],float,Dict[str,float]]]
        # t:float = time.time()
        self.shp = np.shape(Oframe)#[:2]
        preprocess= pupil_preprocess(frame=Oframe,centers=centers, halphsaize=self.halphsize)
        # logging.info(f'pupilNet preprocess time:{time.time()-t}')
        # t:float = time.time()
        lpupil:np.ndarray =  self.buffer.do_inference(preprocess.crops['left'], self.H, self.W)
        rpupil:np.ndarray =  self.buffer.do_inference(preprocess.crops['right'], self.H, self.W)
        # logging.info(f'pupilNet inference time:{time.time()-t}')
        # t:float = time.time()
        postprocess = pupil_postprocess(crops = {'left':lpupil,'right':rpupil},shp = self.shp, ulcs=preprocess.ulcs).info
        return preprocess.crops,postprocess
    
@cython.cclass 
class pupil_preprocess():
    left:Tuple[float,float]
    right:Tuple[float,float]
    frame:np.ndarray
    crops:Dict[str,np.ndarray]
    ulcs:Dict[str,np.ndarray]

    def __init__(self,
                    frame:np.ndarray,
                    centers:Tuple[Tuple[float,float],Tuple[float,float]],
                    halphsaize:int)->None:
        left_crop:np.ndarray 
        right_crop:np.ndarray

        self.left = centers[0]
        self.right = centers[1]
        left_crop,lulc = self.crop(frame=frame, c=centers[0],halphsize=halphsaize)
        right_crop,rulc = self.crop(frame=frame, c=centers[1],halphsize=halphsaize)
        #print(left_crop.shape, right_crop.shape)
        self.crops = {'left':left_crop,'right':right_crop}
        self.ulcs = {'left':lulc,'right':rulc}

    def crop(self,
                frame:np.ndarray,
                c:Tuple[float,float],
                halphsize:int)->Tuple[np.ndarray,Tuple[float,float]]:
        crop:np.ndarray = frame[c[::-1][0]-halphsize:c[::-1][0]+halphsize,
                                c[::-1][1]-halphsize:c[::-1][1]+halphsize]/255
        ulc:Tuple[float,float] = (c[::-1][0]-halphsize,c[::-1][1]-halphsize)
        return crop,ulc

@cython.cclass 
class pupil_postprocess():
    info:Dict[str,Union[Tuple[Tuple[float,float],Tuple[float,float],float],float,Dict[str,float]]]
    scale:List[float]
    def __init__(self,
                    crops:Dict[str,np.ndarray],
                    shp:Tuple[int,int],
                    ulcs:Dict[str, Tuple[float,float]])->None:
        
        tilt:Union(None,float)
        dist:Union(None,float)
        lsize:Union(None,float)
        rsize:Union(None,float)
        
        lellipse:Tuple[Tuple[float,float],Tuple[float,float],float]
        rellipse:Tuple[Tuple[float,float],Tuple[float,float],float]
        self.scale:List[float] = np.divide(shp,crops['left'].shape[:2])[::-1]
        
        try:
            lellipse = self.locate_precise_eyes(crops['left'], ulc = ulcs['left'])
        except Exception as e:
            print('lellipse')
            print(e)
            lellipse = ()
        
        try:
            rellipse = self.locate_precise_eyes(crops['right'], ulc = ulcs['right'])
        except Exception as e:
            print('rellipse')
            print(e)
            rellipse = ()
        #print(lellipse,rellipse)
        if len(lellipse) != 0:
            lsize:float = np.mean(lellipse[1])*pixel2mm*scale #mm
        else:
            lsize = None
        if len(rellipse) != 0:
            rsize:float = np.mean(rellipse[1])*pixel2mm*scale #mm
        else:
            rsize = None
        if all([len(lellipse),len(rellipse)])==0:
            tilt,dist = None,None
        else:
            tilt,dist = self.get_orientation(lellipse[0],rellipse[0])

        self.info = {'left':lellipse, 'right':rellipse,'tilt':tilt,'dist':dist, 'size':{'left':lsize, 'right':rsize}}
        
    def get_orientation(self,
                        left:Tuple[float,float],
                        right:Tuple[float,float])->Tuple[float,float]:
        d:List[float,float] = np.subtract(left,right)
        pdist:float = np.linalg.norm(d)*pixel2mm*scale #mm
        alpha:float = np.rad2deg(np.arctan(d[1]/d[0]))
        return alpha,pdist
    
    def locate_precise_eyes(self,
                            crop:np.ndarray, 
                            ulc:Tuple[float,float])->Union[\
                                                        Tuple[\
                                                        np.ndarray, np.ndarray,Tuple[Tuple[float,float],Tuple[float,float],float]],
                                                        Tuple[None,None,None]]:
        ret:bool
        binary:np.ndarray
        ret,binary = cv2.threshold(crop,0.5,1, cv2.THRESH_BINARY)
        c:List[int]
        if cv2version not in [2,4]:
            _,c,_ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            c,_ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        areas:List[float] = [cv2.contourArea(a) for a in c]
        s:List[cython[float]] = np.sort(areas)
        if len(s)<1:return ()
        cnt:List = c[areas.index(s[-1])]
        ellipse:List = list(cv2.fitEllipse(cnt))
        #logging.info(f'___ {ellipse} ___')

        if ellipse[0] == (0,0): return None, None, None
        ellipse[0] = tuple(np.add(ellipse[0],ulc[::-1]))
        #segmentation = cv2.ellipse(np.zeros_like(binary),ellipse,(255)*3,-1)
        #crop = np.squeeze(crop,-1).astype('float32')
        #masked_image = np.multiply(crop,binary)
        return tuple(ellipse) #masked_image,binary,tuple(ellipse)

@cython.cclass 
class rangeNet():
    engine:trt.tensorrt.ICudaEngine
    buffer:buffer

    def __init__(self,path:str)->None:
        self.engine = load(path).engine
        self.buffer = buffer(self.engine, 1, trt.float32)

        # engine_loader = load(path)
        # engine:trt.tensorrt.ICudaEngine = engine_loader.get_engine(path)
        # self.buffer = buffer(engine, 1, trt.float32)

    def inference(self,
                  Oframe:np.ndarray,
                  centers:Dict[str,Tuple[Tuple[cython.float,cython.float],Tuple[cython.float,cython.float],cython.float]])->np.ndarray:
        # write_log('range '+str(get_linenumber()))
        # t:float = time.time()
        crop:np.ndarray= range_preprocess(frame=Oframe,left = centers['left'][0],right = centers['right'][0]).crop
        # write_log('range '+str(get_linenumber()))
        # t:float = time.time()
        defocus:np.ndarray =  self.buffer.do_inference(crop)
        # logging.info(f'rangeNet inference time:{time.time()-t}')
        # t:float = time.time()
        return range_postprocess(defocus=defocus).defocus
    
@cython.cclass 
class range_preprocess():
    frame:np.ndarray
    left:Tuple[cython.float,cython.float]
    right:Tuple[cython.float,cython.float]
    crop:np.ndarray
    def __init__(self,frame:np.ndarray, left:Tuple[cython.float,cython.float],right:Tuple[cython.float,cython.float])->None:
        if np.shape(frame)[0]==1:
            frame = frame[0]
        self.frame = frame
        self.left=left
        self.right=right
        self.crop = self.pair()

    def pair(self)->np.ndarray:
        #paires left and right eyes in a single crop
        dx:cython.float=0
        dy:cython.float=0
        ulc:List[cython.float,cython.float]
        target:List[cython.float,cython.float]

        ulc = [int(round(min(self.left[1],self.right[1])-150)),int(round(min(self.left[0],self.right[0])-150))]
        target = [int(round(max(self.left[1],self.right[1])+150)),int(round(max(self.left[0],self.right[0])+150))]

        if target[0]>1024:
            dx = target[0]-1024
            ulc[0] = ulc[0]-dx
            target[0]=1024
        
        if target[1]>1280:
            dy = target[1]-1280
            ulc[1] = ulc[1]-dy
            target[1]=1280

        if ulc[0]<0:
            dx = abs(ulc[0])
            ulc[0]=0
            target[0]+=dx
        
        if ulc[1]<0:
            dy = abs(ulc[1])
            ulc[1]=0
            target[1]+=dx

        if len(np.shape(self.frame))==2:
            self.frame = cv2.cvtColor(self.frame,cv2.COLOR_GRAY2BGR)

        return cv2.resize(self.frame[ulc[0]:target[0],ulc[1]:target[1],:],(300,150),interpolation=cv2.INTER_AREA) #width,height 
@cython.cclass    
class range_postprocess():
    defocus:str
    def __init__(self, defocus:np.ndarray)->None:
        labels:List[str]=['unknown','too close','too far','inrange']
        self.defocus = labels[defocus.argmax()]


@cython.cclass    
class ref_preprocess():
    refNet_input:Tuple[np.ndarray,np.ndarray,np.ndarray]
    def __init__(self,crops:List[np.ndarray])->None:

        if len(crops)<32:
            self.refNet_input = self.concat(crops)
        else:
            self.refNet_input = self.concat_6meridians(crops)


    def concat(self,crops:List[np.ndarray])->Tuple[np.ndarray,np.ndarray,np.ndarray]:
        m:np.ndarray
        i:int
        
        m0:List[np.ndarray] = crops[:11]
        M0:np.ndarray = np.zeros((106,106,len(m0)))
        for i,m in enumerate(m0):
            M0[:,:,i] = m.astype('float16')
        # ==========================================
        m60:List[np.ndarray] = crops[11:16]
        m60.append(crops[5])
        m60.extend(crops[16:21])
        #m60 = [imutils.rotate(m,-np.pi/3) for m in m60]
        # concatenate
        M60:np.ndarray = np.zeros((106,106,len(m60)))
        for i,m in enumerate(m60):
            M60[:,:,i] = imutils.rotate(m,np.rad2deg(np.pi/3)).astype('float16')
        # ==========================================
        m120:List[np.ndarray] = crops[21:26]
        m120.append(crops[5])
        m120.extend(crops[26:31])
        #m120 = [imutils.rotate(m,np.pi/3) for m in m120]
        M120:np.ndarray = np.zeros((106,106,len(m120)))
        for i,m in enumerate(m120):
            M120[:,:,i] = imutils.rotate(m,np.rad2deg(-np.pi/3)).astype('float16')
        return M0, M60, M120

    def concat_6meridians(self,crops:List[np.ndarray])->Tuple[np.ndarray,np.ndarray,np.ndarray]:
        m:np.ndarray
        i:int
        # 1==========================================
        m0:List[np.ndarray] = crops[:11]
        M0:np.ndarray = np.zeros((106,106,len(m0)))
        for i,m in enumerate(m0):
            M0[:,:,i] = m.astype('float16')
        # 2==========================================
        m30:List[np.ndarray] = crops[11:15]
        m30.append(crops[5])
        m30.extend(crops[15:19])
        # concatenate
        M30:np.ndarray = np.zeros((106,106,len(m30)))
        for i,m in enumerate(m30):
            M30[:,:,i] = imutils.rotate(m,np.rad2deg(np.pi/3)).astype('float16')
        # 3==========================================
        m60:List[np.ndarray] = crops[19:24]
        m60.append(crops[5])
        m60.extend(crops[24:29])
        # concatenate
        M60:np.ndarray = np.zeros((106,106,len(m60)))
        for i,m in enumerate(m60):
            M60[:,:,i] = imutils.rotate(m,np.rad2deg(np.pi/3)).astype('float16')
        # 4==========================================
        m90:List[np.ndarray] = crops[29:33]
        m90.append(crops[5])
        m90.extend(crops[33:37])
        # concatenate
        M90:np.ndarray = np.zeros((106,106,len(m90)))
        for i,m in enumerate(m90):
            M90[:,:,i] = imutils.rotate(m,np.rad2deg(np.pi/3)).astype('float16')
        # 5==========================================
        m120:List[np.ndarray] = crops[37:42]
        m120.append(crops[5])
        m120.extend(crops[42:47])
        # concatenate
        M120:np.ndarray = np.zeros((106,106,len(m120)))
        for i,m in enumerate(m120):
            M120[:,:,i] = imutils.rotate(m,np.rad2deg(-np.pi/3)).astype('float16')
        # 6==========================================
        m150:List[np.ndarray] = crops[47:51]
        m150.append(crops[5])
        m150.extend(crops[51:])
        #m120 = [imutils.rotate(m,np.pi/3) for m in m120]
        M150:np.ndarray = np.zeros((106,106,len(m150)))
        for i,m in enumerate(m150):
            M150[:,:,i] = imutils.rotate(m,np.rad2deg(-np.pi/3)).astype('float16')
        
        return M0, M30, M60, M90, M120, M150
  
@cython.cclass    
class ref_postprocess():
    r0:float
    r60:float
    r120:float
    ds:float
    dc:float
    ax:float

    def __init__(self,r0:float,r60:float,r120:float)->None:
        self.r0 = r0
        self.r60 = r60
        self.r120 = r120

    def get_refractive_errors(self,tilt:float)->None:

        self.ds = self.round225(self.get_ds(self.r0,self.r60,self.r120)-1)
        self.dc = self.round225(self.get_dc(self.r0,self.r60,self.r120))
        #print('ds:',self.ds,'dc:',self.dc)
        self.ax = self.round225(self.get_ax(self.r0,self.r60,self.r120)-90-tilt)
        if self.ax<0:self.ax+=180
        if self.ax>180:self.ax-=180

    def refEstimator(self,rs:List[float],tilt:float)->None:
        (A,B,D),rsqr = self.estimator(rs)
        self.ax = np.round(np.rad2deg(-np.arctan2(D,B)/2),3)-90-tilt  
        self.dc = np.round(2*np.sqrt(B**2+D**2),3)
        self.ds = np.round(A-np.sqrt(B**2+D**2),3)
        if self.ax<0:self.ax+=180
        if self.ax>180:self.ax-=180
        
    def cofactor(self, n_meridians:int):
        '''
            returns cofactor matrix of equation 6,7,8,.. in paper below:
                 Measurement of Astigmatism by Automated Infrared Photoretinoscopy [Florin Gekeler et.al.-1997]
                 each row consists of derivatives of each equation with respect to coefficients. [1,ci,si]
                 
                 where coefficients are [A,B,D]
        '''
        output = np.zeros((n_meridians,3))
        for row,angle in enumerate(np.arange(0,n_meridians*30,30)):
            output[row,0]=1
            output[row,1]=np.cos(2*angle)
            output[row,2]=np.sin(2*angle)
        return output

    def estimator(self,rs:List[float]):
        '''
            Computes least squares estimation method based on Gauss markov model.
            statistic part and stochastic part are out to return for quality control of computation

            inputs:
                rs: the refractive errors that are computed throgh out refNet.
                the number of refractive errors are correlated with the number of meridians of measurment

            returns:
                estimated coefficients (for this methodology three coefficients are needed). refer to paper below,
                        Measurement of Astigmatism by Automated Infrared Photoretinoscopy [Florin Gekeler et.al.-1997]
                coefficient of determination (R-squared), typically higher value is better. [A,B,D]
        '''
        # pseudoinverse of the cofactor matrix
        cofactor_matrix = self.cofactor(n_meridians=len(rs))
        cofactor_pinv = np.linalg.pinv(cofactor_matrix)
        # Compute the least squares estimate of the coefficients
        coefficients = np.dot(cofactor_pinv, rs)
        # [Stochastic part] Compute the predicted values
        y_pred = np.dot(cofactor_matrix, coefficients)
        # Compute the residuals
        residuals = rs - y_pred
        # Compute Sum of Squared residuals (SSE) and Total Sum of Squares (SST)
        SSE = np.sum(residuals ** 2)
        SST = np.sum((rs - np.mean(rs)) ** 2)
        # [Statistic part] Compute the coefficient of determination (R-squared) 
        R_squared = 1 - (SSE / SST) 
        return coefficients, R_squared

    def transpose(self)->Dict[str,float]:
        self.ds +=self.dc
        self.dc = -1*self.dc
        if self.ax<=90:
            self.ax+=90
        elif self.ax>90:
            self.ax-=90
        return {'ds':self.ds,'dc':self.dc,'ax':self.ax,'r0':self.r0,'r60':self.r60,'r120':self.r120}
        
    def round225(self,x:float)->float: 
        return np.round(x/0.25)*0.25 

    def get_A(self,r0:float,r60:float,r120:float)->float:
        return (r0+r60+r120)/3

    def get_B(self,r0:float,r60:float,r120:float)->float:
        return (-2*r0+r60+r120)/3
    
    def get_D(self,r0:float,r60:float,r120:float)->float:
        return (r60-r120)/np.sqrt(3)
    
    def get_ax(self,r0:float,r60:float,r120:float)->float:
        return np.round(np.rad2deg(-np.arctan2(self.get_D(r0,r60,r120),self.get_B(r0,r60,r120))/2),3)
    
    def get_dc(self,r0:float,r60:float,r120:float)->float:
        return np.round(2*np.sqrt(self.get_B(r0,r60,r120)**2+self.get_D(r0,r60,r120)**2),3)
    
    def get_ds(self,r0:float,r60:float,r120:float)->float:
        return np.round(self.get_A(r0,r60,r120)-np.sqrt(self.get_B(r0,r60,r120)**2+self.get_D(r0,r60,r120)**2),3)

@cython.cclass    
class refNet():
    engine:trt.tensorrt.ICudaEngine
    buffer:buffer
    m:int

    def __init__(self,path:str,multiplier:int=5)->None:
        self.engine = load(path).engine
        self.buffer = buffer(self.engine, 1, trt.float32)
        self.m = multiplier

    def inference(self,
            pupil_info:Tuple[Dict[str,np.ndarray],Dict[str,Union[Tuple[float],float,float,Dict[str,float]]]])->Dict[str,Dict[str,float]]:
        
        # lM0:np.ndarray
        # lM60:np.ndarray
        # lM120:np.ndarray
        # rM0:np.ndarray
        # rM60:np.ndarray
        # rM120:np.ndarray
        # t = time.time()
        left_crops:List[np.ndarray] = [cv2.resize(pi['left'],(150,150),interpolation = cv2.INTER_AREA)[22:-22,22:-22]\
                         for pi,_ in pupil_info]
        right_crops:List[np.ndarray] =[cv2.resize(pi['right'],(150,150),interpolation = cv2.INTER_AREA)[22:-22,22:-22]\
                         for pi,_ in pupil_info]
        left_refNet_inputs = ref_preprocess(crops=left_crops).refNet_input
        left_rs = [self.buffer.do_inference(M)[0]*self.m for M in left_refNet_inputs]

        right_refNet_inputs = ref_preprocess(crops=right_crops).refNet_input
        right_rs = [self.buffer.do_inference(M)[0]*self.m for M in right_refNet_inputs]

        # lM0,lM60,lM120 = ref_preprocess(crops=left_crops).refNet_input
        # rM0,rM60,rM120 = ref_preprocess(crops=right_crops).refNet_input
        # # logging.info(f'refNet preprocess time:{time.time()-t}')
        # # t:float = time.time()
        # lr0:float =  self.buffer.do_inference(lM0)[0]*self.m
        # # logging.info(f'refNet inference m0 l:{time.time()-t}')
        # # t:float = time.time()
        # lr60:float =  self.buffer.do_inference(lM60)[0]*self.m
        # # logging.info(f'refNet inference m60 l:{time.time()-t}')
        # # t:float = time.time()
        # lr120:float =  self.buffer.do_inference(lM120)[0]*self.m
        # # logging.info(f'refNet inference m120 l:{time.time()-t}')
        # # t:float = time.time()
        # rr0:float =  self.buffer.do_inference(rM0)[0]*self.m
        # # logging.info(f'refNet inference m0 r:{time.time()-t}')
        # # t:float = time.time()
        # rr60:float =  self.buffer.do_inference(rM60)[0]*self.m
        # # logging.info(f'refNet inference m60 r:{time.time()-t}')
        # # t:float = time.time()
        # rr120:float =  self.buffer.do_inference(rM120)[0]*self.m
        # logging.info(f'refNet inference m120 r:{time.time()-t}')
        # t:float = time.time()
        lp:ref_postprocess = ref_postprocess(lr0,lr60,lr120)
        #print('pupil_info:',pupil_info)
        lp.get_refractive_errors(pupil_info[5][1]['tilt'])
        left:Dict[str,float] = lp.transpose()
        rp:ref_postprocess = ref_postprocess(rr0,rr60,rr120)
        rp.get_refractive_errors(pupil_info[5][1]['tilt'])
        right:Dict[str,float] = rp.transpose()
        # logging.info(f'refNet postprocess time:{time.time()-t}')
        return {'left':left,'right':right}

@cython.cclass
class infoClass():
    frame:np.ndarray
    centers:Tuple[Tuple[float,float],Tuple[float,float]]
    Range:str




