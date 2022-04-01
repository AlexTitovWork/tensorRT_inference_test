#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <sys/stat.h>
#include <unordered_set>
#include <vector>
#include "cuda_runtime_api.h"

#include "NvInfer.h"
#include "NvUtils.h"
#include "NvInferRuntime.h"

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "sampleEngines.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>


// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
// for cuda::divide support operation
#include <opencv2/cudaarithm.hpp>


using namespace nvinfer1;
using namespace std;
using namespace cv;


// using namespace std;
// static Logger gLogger;
class Logger : public nvinfer1::ILogger           
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

// class Logger : public nvinfer1::ILogger
// {
// public:
//     void log(Severity severity, const char* msg) override {
//         // remove this 'if' if you need more logged info
//         if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR) || (severity <= Severity::kWARNING)) {
//             std::cout << msg << "n";
//         }
//     }
// } gLogger;


size_t getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}

std::vector< std::string > getClassNames(const std::string& imagenet_classes)
{
    std::ifstream classes_file(imagenet_classes);
    std::vector< std::string > classes;
    if (!classes_file.good())
    {
        std::cerr << "ERROR: can't read file with classes names.n";
        return classes;
    }
    std::string class_name;
    while (std::getline(classes_file, class_name))
    {
        classes.push_back(class_name);
    }
    return classes;
}

// destroy TensorRT objects if something goes wrong
struct TRTDestroy
{
    template< class T >
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

template< class T >
using TRTUniquePtr = std::unique_ptr< T, TRTDestroy >;


void preprocessImage(const std::string& image_path, float* gpu_input, const nvinfer1::Dims& dims)
{
    cv::Mat frame = cv::imread(image_path);
    if (frame.empty())    {
        std::cerr << "Input image " << image_path << " load failed\n";
        return;
    }
    else {
        std::cout<<image_path<< " image loaded successfully !" <<std::endl;
    }
    cv::cuda::GpuMat gpu_frame;
    // upload image to GPU
    gpu_frame.upload(frame);
    // gpu_frame.download(frame);
    std::cout<<"gpu_frame uploaded ok!"<<std::endl;
    auto input_width = dims.d[2];
    auto input_height = dims.d[1];
    auto channels = dims.d[0];
    auto input_size = cv::Size(input_width, input_height);
    // resize
    cv::cuda::GpuMat resized;
    cv::cuda::resize(gpu_frame, resized, input_size, 0, 0, cv::INTER_NEAREST);
    std::cout<<"cv::cuda::resized ok!"<<std::endl;

    // cv::resize(gpu_frame, resized, input_size, 0, 0, cv::INTER_NEAREST);

    cv::cuda::GpuMat flt_image;
    resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
    cv::cuda::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);
    // cv::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);

    cv::cuda::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);
    // cv::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);

    std::vector< cv::cuda::GpuMat > chw;
    for (size_t i = 0; i < channels; ++i){
        chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, gpu_input + i * input_width * input_height));
    }
    cv::cuda::split(flt_image, chw);
    // cv::split(flt_image, chw);

}




void postprocessResults(float *gpu_output, const nvinfer1::Dims &dims, int batch_size)
{
    // get class names
    auto classes = getClassNames("./data/imagenet_classes.txt");

    // copy results from GPU to CPU
    std::vector< float > cpu_output(getSizeByDim(dims) * batch_size);
    cudaMemcpy(cpu_output.data(), gpu_output, cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    // calculate softmax
    std::transform(cpu_output.begin(), cpu_output.end(), cpu_output.begin(), [](float val) {return std::exp(val);});
    auto sum = std::accumulate(cpu_output.begin(), cpu_output.end(), 0.0);
    // find top classes predicted by the model
    std::vector< int > indices(getSizeByDim(dims) * batch_size);
    // generate sequence 0, 1, 2, 3, ..., 999
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&cpu_output](int i1, int i2) {return cpu_output[i1] > cpu_output[i2];});
    // print results
    int i = 0;
    while (cpu_output[indices[i]] / sum > 0.005)
    {
        if (classes.size() > indices[i])
        {
            std::cout << "class: " << classes[indices[i]] << " | ";
        }
        std::cout << "confidence: " << 100 * cpu_output[indices[i]] / sum << "% | index: " << indices[i] << "\n";
        ++i;
    }
}



int main(void){

    // Create and save engine
    // std::string name_engine = "/home/interceptor/Документы/Git_Medium_repo/Binary_search_engine_CUDA/tensorRT/tensorRT_pytorch_to_onxx/data/resnet50_engine.trt";
    std::string name_engine = "./data/resnet50_engine.trt";
    // std::string image_path = "./data/coffee_cup1.jpg";
    std::string image_path = "./data/coffee_cup2.jpg";

    bool create_engine = true;

    // cout << "Loading TensorRT engine from plan file..." << endl;
    // sample::gLogInfo << "Loading TensorRT engine from plan file " << name_engine << std::endl;

    std::ifstream planFile(name_engine);

    if (!planFile.is_open())
    {
        //  sample::gLogError << "Could not open plan file: " << name_engine << std::endl;
         std::cout << "Could not open plan file: " << name_engine << std::endl;

        return -1;
    }

    std::stringstream planBuffer;
    planBuffer << planFile.rdbuf();
    std::string plan = planBuffer.str();
    IRuntime *runtime = createInferRuntime(logger);
    

    // IRuntime* runtime = createInferRuntime(gLogger);

    
    ICudaEngine *engine = runtime->deserializeCudaEngine((void *)plan.data(), plan.size(), nullptr);
    int batch_size = 1;
    
    if (engine != NULL){
        
        std::cout<<"Engine loaded successfully!"<<std::endl;

        IExecutionContext *context = engine->createExecutionContext();
        std::vector< nvinfer1::Dims > input_dims; // we expect only one input
        std::vector< nvinfer1::Dims > output_dims; // and one output
        std::vector< void* > buffers(engine->getNbBindings()); // buffers for input and output data
        for (size_t i = 0; i < engine->getNbBindings(); ++i)
        {
            auto binding_size = getSizeByDim(engine->getBindingDimensions(i)) * batch_size * sizeof(float);
            cudaMalloc(&buffers[i], binding_size);
            if (engine->bindingIsInput(i))
            {
                input_dims.emplace_back(engine->getBindingDimensions(i));
            }
            else
            {
                output_dims.emplace_back(engine->getBindingDimensions(i));
            }
        }
        if (input_dims.empty() || output_dims.empty())
        {
            std::cerr << "Expect at least one input and one output for networkn";
            return -1;
        }

                // preprocess input data

        preprocessImage(image_path, (float*)buffers[0], input_dims[0]);
        std::cout << "Preprocessing of image ok! " << name_engine << std::endl;

        // inference
        context->enqueue(batch_size, buffers.data(), 0, nullptr);
        std::cout << "context->enqueue(batch_size, buffers.data(), 0, nullptr) of data ok! " << name_engine << std::endl;

        // post-process results
        postprocessResults((float *) buffers[1], output_dims[0], batch_size);

        std::cout<<"Processing of image ok!"<<std::endl;
        // for (void* buf : buffers)
        // {
        //     cudaFree(buf);

        //     context->destroy();
        //     engine->destroy();
        //     runtime->destroy();
        // }
        return 0;
    }
    else {
        return -1;
    }
}