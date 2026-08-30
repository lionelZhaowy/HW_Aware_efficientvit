import torch
from mqbench_export.prepare_by_platform import BackendType

# 统一的后端映射字典
BackendMap = {
    'tensorrt': BackendType.Tensorrt,
    'nnie': BackendType.NNIE,
    'ppl': BackendType.PPLW8A16,
    'snpe': BackendType.SNPE,
    'vitis': BackendType.Vitis,
    'tengine_u8': BackendType.Tengine_u8
}

# 统一的自定义量化配置字典
prepare_custom_config_dict = dict(
    extra_qconfig_dict={
        'w_observer': 'MinMaxObserver',
        'a_observer': 'EMAMinMaxObserver',
        'w_fakequantize': 'LearnableFakeQuantize',
        'a_fakequantize': 'LearnableFakeQuantize',
        'w_qscheme': {
            'bit': 8,
            'symmetry': True,
            'per_channel': False,
            'pot_scale': False
        },
        'a_qscheme': {
            'bit': 8,
            'symmetry': True,
            'per_channel': False,
            'pot_scale': False
        }
    },
    extra_quantizer_dict={
        'additional_node_name': ['input', 'output'],
        'additional_function_type': [
            torch.cat, torch.add, torch.matmul, torch.sigmoid, torch.softmax
        ],
        'additional_module_type': (
            torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Softmax, torch.nn.Linear, torch.nn.LayerNorm
        ),
    }
)

def get_quant_config():
    """可选：提供一个获取配置的函数，方便后续扩展（如根据不同模型返回不同配置）"""
    return BackendMap, prepare_custom_config_dict
