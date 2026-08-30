import os
import sys
sys.path.append(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
from efficientvit.cls_model_zoo import create_efficientvit_cls_model
from efficientvit.models.utils import load_state_dict_from_file

from mqbench_export.prepare_by_platform import prepare_by_platform
from mqbench_export.convert_deploy import convert_deploy
from applications.quant_config import BackendMap, prepare_custom_config_dict


if __name__ == '__main__':
    model_name = 'efficientvit-b1'
    deploy_weight_path = './logs/efficientvit_cls/efficientvit_b1_QAT/checkpoint/model_best.pt'
    onnx_dir = './onnx'
    onnx_prefix = 'EfficientViT_b0'
    image_size = 256

    quant_backend = BackendMap['tensorrt']
    model = create_efficientvit_cls_model(model_name, pretrained=False, dropout=0.05)
    model = prepare_by_platform(model, quant_backend, prepare_custom_config_dict)
    weight = load_state_dict_from_file(deploy_weight_path)
    model.load_state_dict(weight)
    model.cpu()
    model.eval()
    if os.path.exists(onnx_dir) is False:
        os.makedirs(onnx_dir)
    convert_deploy(
        model,
        quant_backend,
        input_shape_dict={'data': [1, 3, image_size, image_size]},
        output_path=onnx_dir,
        model_name=onnx_prefix,
        input_names=['input'],
        output_names=['output'],
    )
