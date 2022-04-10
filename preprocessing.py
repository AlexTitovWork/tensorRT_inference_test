import cv2
import torch
import onnx
from albumentations import Resize, Compose
from albumentations.pytorch.transforms import  ToTensor
from albumentations.augmentations.transforms import Normalize
from torchvision import models


# Prepare image for model test 
def preprocess_image(img_path):
    # transformations for the input data
    transforms = Compose([
        Resize(224, 224, interpolation=cv2.INTER_NEAREST),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensor(),
    ])
    
    # read input image
    input_img = cv2.imread(img_path)
    # do transformations
    input_data = transforms(image=input_img)["image"]
    batch_data = torch.unsqueeze(input_data, 0)
    return batch_data

# Get human readeable format
def postprocess(output_data):
    # get class names
    # with open("/home/interceptor/Документы/Git_Medium_repo/Binary_search_engine_CUDA/tensorRT/tensorRT_pytorch_to_onxx/data/imagenet_classes.txt") as f:
    with open("./data/imagenet_classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]

    result = []
    # calculate human-readable value by softmax
    confidences = torch.nn.functional.softmax(output_data, dim=1)[0] * 100
    # find top predicted classes
    _, indices = torch.sort(output_data, descending=True)
    i = 0
    # print the top classes predicted by the model
    while confidences[indices[0][i]] > 0.5:
        class_idx = indices[0][i]
        print(
            "class:",
            classes[class_idx],
            "confidence:",
            confidences[class_idx].item(),
            "index:",
            class_idx.item(),
        )
        # Added return data for server version
        result.append(["class:", classes[class_idx],
            "confidence:", confidences[class_idx].item(),
            "index:", class_idx.item()])

        i += 1
    return result


def main_test():

    model = models.resnet50(pretrained=True)

    input = preprocess_image("/home/interceptor/Документы/Git_Medium_repo/Binary_search_engine_CUDA/tensorRT/tensorRT_pytorch_to_onxx/data/coffee_cup1.jpg").cuda()
    model.eval()
    model.cuda()
    output = model(input)
    postprocess(output)
    # And results:
    # class: cup, confidence: 92.430747%, index: 968
    # class: espresso, confidence: 6.138075%, index: 967
    # class: coffee mug, confidence: 0.728557%, index: 504


    # Converter to ONXX
    # 2. Convert the PyTorch model to ONNX format

    ONNX_FILE_PATH = './tensorRT_pytorch_to_onxx/data/resnet50.onnx'
    torch.onnx.export(model, input, ONNX_FILE_PATH, input_names=['input'],
                    output_names=['output'], export_params=True)
    # To check that the model converted fine, call onnx.checker.check_model:
    onnx_model = onnx.load(ONNX_FILE_PATH)
    # onnx.checker.check_model(onnx_model) 
    # TODO debug checkers

    print("Done")

    # visualisation of model from onnx
    # python3 -m pip install netron

    #issue 
    # Traceback (most recent call last):
    #   File "/home/interceptor/Документы/Git_Medium_repo/Binary_search_engine_CUDA/tensorRT/tensorRT_pytorch_to_onxx/preprocessing.py", line 71, in <module>
    #     onnx.checker.check_model(onnx_model)
    #   File "/home/interceptor/.local/lib/python3.6/site-packages/onnx/checker.py", line 86, in check_model
    #     C.check_model(model.SerializeToString())
    # onnx.onnx_cpp2py_export.checker.ValidationError: Your model ir_version is higher than the checker's.
    # interceptor@interceptor-N750JK:~/Документы/Git_Medium_repo/Binary_search_engine_CUDA/tensorRT$ pip install --user --upgrade onnx1

# main_test()