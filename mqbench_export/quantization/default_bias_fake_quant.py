from mqbench_export.fake_quantize import TqtFakeQuantize
from mqbench_export.observer import MinMaxFloorObserver
from mqbench_export.scheme import QuantizeScheme


bias_fakeq_param = {}
bias_qscheme = QuantizeScheme(symmetry=True, per_channel=False, pot_scale=True, bit=8)
b_fakequantize = TqtFakeQuantize
b_qconfig = b_fakequantize.with_args(observer=MinMaxFloorObserver, **bias_fakeq_param, **bias_qscheme.to_observer_params())
bias_fake_quantizer = b_qconfig
