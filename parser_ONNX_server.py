from html5lib import serialize
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import torch


from preprocessing import preprocess_image
from preprocessing import postprocess
import json
from pprint import pprint
from flask import Flask, request, jsonify
# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()



class ServerInstantTensorRT:
    engine = []
    context = []
    runtime = []
    device_input = []
    host_output = []
    device_output = []
    def __init__(self):

        print(trt.__version__)
        
        # 1. First phase Build engine from ONNX format
        # initialize TensorRT engine and parse ONNX model
        # ONNX_FILE_PATH = "/home/interceptor/Документы/Git_Medium_repo/Binary_search_engine_CUDA/tensorRT/tensorRT_pytorch_to_onxx/data/resnet50.onnx"
        # 
        # ONNX_FILE_PATH = "./data/resnet50.onnx"
        # engine, context = build_engine(ONNX_FILE_PATH)
        
        # 2. Second phase serialize engine 
        # Serialize engine and save it to the drive
        # 
        # build_and_serialized_engine(engine)

        # 3. Third phase load engine from drive.

        # Call internal self.engine, self.context, self.runtime
        self.engine, self.context, self.runtime = self.load_and_deserialized_engine()

        # get sizes of input and output and allocate memory required for input data and for output data
        for self.binding in self.engine:
            if self.engine.binding_is_input(self.binding):  # we expect only one input
                input_shape = self.engine.get_binding_shape(self.binding)
                input_size = trt.volume(input_shape) *self.engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
                self.device_input = cuda.mem_alloc(input_size)
            else:  # and one output
                self.output_shape = self.engine.get_binding_shape(self.binding)
                # create page-locked memory buffers (i.e. won't be swapped to disk)
                self.host_output = cuda.pagelocked_empty(trt.volume(self.output_shape) * self.engine.max_batch_size, dtype=np.float32)
                self.device_output = cuda.mem_alloc(self.host_output.nbytes)
        # Create a stream in which to copy inputs/outputs and run inference.
        self.stream = cuda.Stream()

        ## Use call method insted bottom code, may call from client side
        # self.image_classify()

        # # Moved this part to the server listener side
        #     # preprocess input data
        # # host_input = np.array(preprocess_image("/home/interceptor/Документы/Git_Medium_repo/Binary_search_engine_CUDA/tensorRT/tensorRT_pytorch_to_onxx/data/coffee_cup1.jpg").numpy(), dtype=np.float32, order='C')
        # host_input = np.array(preprocess_image("./data/coffee_cup1.jpg").numpy(), dtype=np.float32, order='C')

        # # host_input = np.array(preprocess_image("/home/interceptor/Документы/Git_Medium_repo/Binary_search_engine_CUDA/tensorRT/tensorRT_pytorch_to_onxx/data/sofa.jpeg").numpy(), dtype=np.float32, order='C')

        # cuda.memcpy_htod_async(device_input, host_input, stream)
        # # run inference
        # context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
        # cuda.memcpy_dtoh_async(host_output, device_output, stream)
        # stream.synchronize()

        # # postprocess results
        # # output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, output_shape[0],output_shape[1])
        # output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, output_shape[1])
        # postprocess(output_data)
    
    def image_classify(self):
        print("Method in call image_classify() started...")
        
        # Moved this part to the server listener side
        # preprocess input data
        # host_input = np.array(preprocess_image("/home/interceptor/Документы/Git_Medium_repo/Binary_search_engine_CUDA/tensorRT/tensorRT_pytorch_to_onxx/data/coffee_cup1.jpg").numpy(), dtype=np.float32, order='C')
        self.host_input = np.array(preprocess_image(
            "./data/coffee_cup1.jpg").numpy(), dtype=np.float32, order='C')
        # host_input = np.array(preprocess_image("/home/interceptor/Документы/Git_Medium_repo/Binary_search_engine_CUDA/tensorRT/tensorRT_pytorch_to_onxx/data/sofa.jpeg").numpy(), dtype=np.float32, order='C')
        cuda.memcpy_htod_async(self.device_input, self.host_input, self.stream)
        # run inference
        self.context.execute_async(bindings=[int(self.device_input), int(self.device_output)], stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_output, self.device_output, self.stream)
        self.stream.synchronize()

        # postprocess results
        # output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, output_shape[0],output_shape[1])
        output_data = torch.Tensor(self.host_output).reshape(
            self.engine.max_batch_size, self.output_shape[1])

        postprocess(output_data)


    def build_engine(self, onnx_file_path):
        # initialize TensorRT engine and parse ONNX model
        builder = trt.Builder(TRT_LOGGER)
        mode = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(mode)
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # parse ONNX
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            parser.parse(model.read())
        print('Completed parsing of ONNX file')

        
        
        # Depricated for new version TRT , but this code worked on GTX850 
        # allow TensorRT to use up to 1GB of GPU memory for tactic selection
        # builder.max_workspace_size = 1 << 30

        #New API in 8.XX TensoarRT
        config = builder.create_builder_config()
        workspace = 4 
        # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 20) # fix TRT 8.4 deprecation notice , use 4 MB workspase
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30) # fix TRT 8.4 deprecation notice, use 4 GB workspase


        # we have only one image in batch
        builder.max_batch_size = 1
        # use FP16 mode if possible
        if builder.platform_has_fast_fp16:
            # for GTX 8050m  
            # builder.fp16_mode = True
            # https://github.com/NVIDIA/TensorRT/issues/1820 on RTX3090
            config.set_flag(trt.BuilderFlag.FP16)


        # generate TensorRT engine optimized for the target platform
        print('Building an engine...')
        # https://forums.developer.nvidia.com/t/build-engine-error-when-use-pointnet-like-structure-and-tensorrt-8-0-1-6/183569/8?u=alexeytitovwork
        engine = builder.build_engine(network,config )
        context = engine.create_execution_context()
        print("Completed creating Engine")

        return engine, context

    '''
    Serialize engine and save it to the drive
    '''
    def build_and_serialized_engine(self,engine):

        serialized_engine = engine.serialize();
        with open("./data/serialized_engine_python.trt", "wb") as f:
            f.write(serialized_engine)
    '''
    Load and deserialization engine to runtime
    '''    
    def load_and_deserialized_engine(self):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open("./data/serialized_engine_python.trt", "rb") as f:
            serialized_engine = f.read()
        
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        
        print("deserialize_cuda_engine loaded ok!")
        context = engine.create_execution_context()
        print("create_execution_context created ok!")

        return engine, context, runtime

# BUG
# [04/10/2022-02:14:09] [TRT] [E] 1: [convolutionRunner.cpp::execute::211] Error Code 1: Cask (Cask convolution execution)
# [04/10/2022-02:14:10] [TRT] [E] 1: [checkMacros.cpp::catchCudaError::271] Error Code 1: Cuda Runtime (invalid resource handle)
# FIX
# Starting the Flask application in non-threaded mode worked for me
# https://stackoverflow.com/questions/49595175/pycuda-context-error-when-using-flask

app = Flask(__name__)
with app.app_context():

    server = ServerInstantTensorRT()
    print(server)
    print("ResNet50 started on the Flask Server ok!")

@app.route('/image_classify', methods=['POST'])
def load_image():
    server.image_classify()
    return "OK"

@app.route('/get_classes', methods=['GET'])
def parse_request():
    # data = request.data  # data is empty
    result = server.image_classify()
    print(result)
    return jsonify(result)

if __name__ == '__main__':
    # app.run()
    app.run(debug=True, use_reloader=False, threaded=False) # FIX

