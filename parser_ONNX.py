from html5lib import serialize
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import torch


from preprocessing import preprocess_image
from preprocessing import postprocess


# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()

def build_engine(onnx_file_path):
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
def build_and_serialized_engine(engine):

    serialized_engine = engine.serialize();
    with open("./data/serialized_engine_python.trt", "wb") as f:
        f.write(serialized_engine)
'''
Load and deserialization engine to runtime
'''    
def load_and_deserialized_engine():
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open("./data/serialized_engine_python.trt", "rb") as f:
        serialized_engine = f.read()
    
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    
    print("deserialize_cuda_engine loaded ok!")
    context = engine.create_execution_context()
    print("create_execution_context created ok!")

    return engine, context, runtime


if __name__ == '__main__':
    print(trt.__version__)
    
    # 1. First phase Build engine from ONNX format
    # initialize TensorRT engine and parse ONNX model
    # ONNX_FILE_PATH = "/home/interceptor/Документы/Git_Medium_repo/Binary_search_engine_CUDA/tensorRT/tensorRT_pytorch_to_onxx/data/resnet50.onnx"
    ONNX_FILE_PATH = "./data/resnet50.onnx"
    
    engine, context = build_engine(ONNX_FILE_PATH)
    
    # 2. Second phase serialize engine 
    # Serialize engine and save it to the drive
    build_and_serialized_engine(engine)

    # 3. Third phase load engine from drive.

    # engine, context, runtime = load_and_deserialized_engine()

    # get sizes of input and output and allocate memory required for input data and for output data
    for binding in engine:
        if engine.binding_is_input(binding):  # we expect only one input
            input_shape = engine.get_binding_shape(binding)
            input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
            device_input = cuda.mem_alloc(input_size)
        else:  # and one output
            output_shape = engine.get_binding_shape(binding)
            # create page-locked memory buffers (i.e. won't be swapped to disk)
            host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
            device_output = cuda.mem_alloc(host_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()

        # preprocess input data
    host_input = np.array(preprocess_image("/home/interceptor/Документы/Git_Medium_repo/Binary_search_engine_CUDA/tensorRT/tensorRT_pytorch_to_onxx/data/coffee_cup1.jpg").numpy(), dtype=np.float32, order='C')
    # host_input = np.array(preprocess_image("/home/interceptor/Документы/Git_Medium_repo/Binary_search_engine_CUDA/tensorRT/tensorRT_pytorch_to_onxx/data/sofa.jpeg").numpy(), dtype=np.float32, order='C')

    cuda.memcpy_htod_async(device_input, host_input, stream)
    # run inference
    context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_output, device_output, stream)
    stream.synchronize()

    # postprocess results
    # output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, output_shape[0],output_shape[1])
    output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, output_shape[1])

    postprocess(output_data)



