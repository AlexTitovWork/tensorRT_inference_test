# Project tensorRT_inference_test
The tensor RT inference test of resnet50 classifyer. 

./trtexec --onnx=./data/resnet50.onnx --saveEngine=./data/resnet50_engine.trt


<!-- ADDED  -->
Flask service with low time inference ResNet50 model.
- model use converted TensoRT model "data/serialized_engine_python.trt", that use only on target mashine.
This model not portable on other machine. We need creating it from ONNX model in to target GPU arhitechture. (if model converting on RTX3090 Ti, it work only on this machine with 3090Ti)
- script for converting ONNX model available in "usr/bit/tensort/bin" with standart instalation the TensorRT engine.
- *.trt model placed in "data\" directory of project.
- For prepare FaaS (Function ResNet50 as service on Flask web framework) use:
  "pip install -r requirenments.txt" 
  for install all dependency
- For start Service on Flask use:
  "python3 tensorRT_inference_test/parser_ONNX_server.py" 
- For test image loading by POST request and get inference from ResNet50 DNN classifier (FaaS) try:
  "time python /tensorRT/tensorRT_inference_test/parser_ONNX_client.py"
- For monitoring memory usage of GPU:
  "nvitop -m"
  -------------------------------------------------------------------
- For test Flask endpoint availability use small test like this:
"time curl -X GET -H "Content-type: applicat
ion/json" -d "{""}" "http://127.0.0.1:5000/get_classes""
- Test result of Flask service:
    "8.4.0.6
    deserialize_cuda_engine loaded ok!
    create_execution_context created ok!
    <__main__.ServerInstantTensorRT object at 0x7f47ed43c5c0>
    ResNet50 started on the Flask Server ok!
    * Serving Flask app "parser_ONNX_server" (lazy loading)
    * Environment: production
    WARNING: This is a development server. Do not use it in a production deployment.
    Use a production WSGI server instead.
    * Debug mode: on
    * Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)
    Method in call image_classify() started...
    class: 968: 'cup', confidence: 57.08656311035156 index: 968
    class: 967: 'espresso', confidence: 41.59719467163086 index: 967
    class: 504: 'coffee mug', confidence: 0.9361387491226196 index: 504
    127.0.0.1 - - [13/Apr/2022 23:12:08] "GET /get_classes HTTP/1.1" 200 -"

- Test client result:
        "time curl -X GET -H "Content-type: applicat
    ion/json" -d "{""}" "http://127.0.0.1:5000/get_classes"
    [["class:", "968: 'cup',", "confidence:", 57.08656311035156, "index:", 968], ["class:", "967: 'espresso',", "confidence:", 41.59719467163086, "index:", 967], ["class:", "504: 'coffee mug',", "confidence:", 0.9361387491226196, "index:", 504]]
    real    0m11,992s
    user    0m0,012s
    sys     0m0,013s"
- Note. Using MPS not improove inference with TensorRT+Flask+Python, try to test.
  "nvidia-cuda-mps-control -f" 