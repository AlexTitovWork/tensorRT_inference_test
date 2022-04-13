# tensorRT_inference_test
The tensor RT inference test of resnet50 classifyer. 

./trtexec --onnx=./data/resnet50.onnx --saveEngine=./data/resnet50_engine.trt


<!-- ADDED  -->
FLask service with low time inference ResNet50 model.
- model use converted TensoRT model "data/serialized_engine_python.trt", that use only on target mashine.
this model not portable on other machine. We need creating it from ONNX model in to target GPU arhitechture. (if model converting on RTX3090 Ti, it worked only on this machine with 3090Ti)
- script for converting ONNX model available in "usr/bit/tensort/bin" with standart instalation on TensorRT engine.
- *.trt model placed in "data\" directory of project.
- For start FaaS (Function ResNet50 as service on Flask web framework) use:
  "pip install -r requirenments.txt" 
  for install all dependency
- For start Service on Flask use:
  "python3 tensorRT_inference_test/parser_ONNX_server.py" 
- For test image loading by POST request and get inference from ResNet50 DNN classifier (FaaS) try:
  "time python /tensorRT/tensorRT_inference_test/parser_ONNX_client.py"
- For control memory using of GPU:
  "nvitop -m"
- For test Flask endpoint availability use small test like this:
"time curl -X GET -H "Content-type: application/json" -d "{""}" "http://127.0.0.1:5000/get_classes""