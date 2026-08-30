import torch

from mqbench.utils.logger import logger


def set_fixed_scale(model, param_name='x_post_act_fake_quantizer.scale', value=[0.0078125]):
    """
    将模型中指定名称的参数设置为固定值，并冻结其梯度。
    
    参数:
        model (torch.nn.Module): 目标模型。
        param_name (str): 需要设置的参数名称，如 'x_post_act_fake_quantizer.scale'。
        value (float): 参数的新值。
    返回:
        bool: 如果参数存在并成功设置返回True，否则返回False。
    """
    param_name = param_name if param_name.endswith('.scale') else (param_name + '.scale')
    # 遍历模型的所有参数
    for name, param in model.named_parameters():
        if name == param_name:
            # 设置参数值，确保数值类型和设备与原参数一致
            param.data = torch.tensor(value, dtype=param.dtype, device=param.device)
            # 冻结参数，防止在训练中被优化
            param.requires_grad = False
            logger.info(f"成功设置参数 {param_name} 的值为 {value}，并冻结梯度。")
            return True
    
    logger.info(f"警告：未找到参数 {param_name}。")
    return False


def set_fixed_scale_calibrate(model, param_name='x_post_act_fake_quantizer', value=[0.0078125]):
    """
    将模型中指定名称的参数设置为固定值，并冻结其梯度。
    
    参数:
        model (torch.nn.Module): 目标模型。
        param_name (str): 需要设置的参数名称，如 'x_post_act_fake_quantizer'。
        value (float): 参数的新值。
    返回:
        bool: 如果参数存在并成功设置返回True，否则返回False。
    """
    scale_name = param_name if param_name.endswith('.scale') else (param_name + '.scale')
    # 遍历模型的所有参数
    for name, param in model.named_parameters():
        if name == scale_name:
            # 设置参数值，确保数值类型和设备与原参数一致
            param.data = torch.tensor(value, dtype=param.dtype, device=param.device)
            # 冻结参数，防止在训练中被优化
            param.requires_grad = False
            logger.info(f"成功设置参数 {param_name} 的值为 {value}，并冻结梯度。")
        
    logger.info('Enable observer and Disable quantize.')
    module_name = param_name.strip('.scale') if param_name.endswith('.scale')  else param_name
    for name, submodule in model.named_modules():
        if name == module_name:
            logger.info('Enable observer and Disable quant: {}'.format(name))
            submodule.disable_observer()
            submodule.disable_fake_quant()

def enable_calibration(model):
    logger.info('Enable observer and Disable quantize.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            logger.debug('Enable observer and Disable quant: {}'.format(name))
            submodule.enable_observer()
            submodule.disable_fake_quant()


def enable_calibration_woquantization(model, quantizer_type='fake_quant'):
    logger.info('Enable observer and Disable quantize for {}'.format(quantizer_type))
    for name, submodule in model.named_modules():
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            if quantizer_type not in name: 
                submodule.disable_observer()
                submodule.disable_fake_quant()
                continue
            logger.debug('Enable observer and Disable quant: {}'.format(name))
            submodule.enable_observer()
            submodule.disable_fake_quant()


def enable_calibration_quantization(model, quantizer_type='fake_quant'):
    logger.info('Enable observer and Enable quantize for {}'.format(quantizer_type))
    for name, submodule in model.named_modules():
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            if quantizer_type not in name: 
                submodule.disable_observer()
                submodule.disable_fake_quant()
                continue
            logger.debug('Enable observer and Enable quant: {}'.format(name))
            submodule.enable_observer()
            submodule.enable_fake_quant()


def enable_quantization(model, weight_cali_on=False, act_cali_on=False):
    '''
    We enable all quantization for quantization aware training.
    But we sometimes remain weight calibration on for update minmax all along.
    For some hardware, there is no weight quant param to be set, which mean it will calculate
    min / max for weight.
    Assume weight scale * 127 > abs(weight).max() after some training. Training scale and deploy
    scale can be various, so we have to update range every iter.
    '''
    logger.info('Disable observer and Enable quantize.')
    if weight_cali_on:
        logger.info('Enable observer for weight.')
    if act_cali_on:
        logger.info('Enable observer for activation.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            submodule.enable_fake_quant()
            if weight_cali_on and 'weight_fake_quant' in name:
                logger.debug('Enable observer and Enable quant: {}'.format(name))
                submodule.enable_observer()
            elif act_cali_on and 'act_fake_quant' in name:
                logger.debug('Enable observer and Enable quant: {}'.format(name))
                submodule.enable_observer()
            else:
                logger.debug('Disable observer and Enable quant: {}'.format(name))
                submodule.disable_observer()


def disable_all(model):
    logger.info('Disable observer and Disable quantize.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            logger.debug('Disable observer and Disable quantize: {}'.format(name))
            submodule.disable_observer()
            submodule.disable_fake_quant()


def enable_all(model):
    '''Enable calibration and quantization for every iter, means min / max can be updated
    while training. Use for QAT but can not set range.
    '''
    logger.info('Enable observer and Enable quantize.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            logger.debug('Enable observer and Enable quantize: {}'.format(name))
            submodule.enable_observer()
            submodule.enable_fake_quant()
