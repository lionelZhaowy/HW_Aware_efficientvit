import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn

from efficientvit.models.efficientvit.backbone import EfficientViTBackbone, EfficientViTLargeBackbone
from efficientvit.models.nn import ConvLayer, LinearLayer, OpSequential
from efficientvit.models.utils import build_kwargs_from_config

__all__ = [
    "EfficientViTCls",
    ######################
    "efficientvit_cls_b0",
    "efficientvit_cls_b1",
    "efficientvit_cls_b2",
    "efficientvit_cls_b3",
    ######################
    "efficientvit_cls_l1",
    "efficientvit_cls_l2",
    "efficientvit_cls_l3",
]


class ClsHead(OpSequential):
    def __init__(
        self,
        in_channels: int,
        width_list: list[int],
        # n_classes=1000,
        n_classes=100,
        dropout=0.0,
        norm="bn2d",
        # act_func="hswish",
        act_func="relu",
        fid="stage_final",
    ):
        ops = [
            ConvLayer(in_channels, width_list[0], 1, norm=norm, act_func=act_func),
            nn.AdaptiveAvgPool2d(output_size=1),
            # LinearLayer(width_list[0], width_list[1], False, norm="ln", act_func=act_func),
            LinearLayer(width_list[0], width_list[1], False, dropout, norm='bn1d', act_func=act_func),
            LinearLayer(width_list[1], n_classes, True, dropout, None, None),
        ]
        super().__init__(ops)

        self.fid = fid

    def forward(self, feed_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        x = feed_dict[self.fid]
        return OpSequential.forward(self, x)


class EfficientViTCls(nn.Module):
    def __init__(self, backbone: EfficientViTBackbone | EfficientViTLargeBackbone, head: ClsHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feed_dict = self.backbone(x)
        output = self.head(feed_dict)
        return output


def efficientvit_cls_b0(**kwargs) -> EfficientViTCls:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b0

    backbone = efficientvit_backbone_b0(**kwargs)

    head = ClsHead(
        in_channels=128,
        width_list=[1024, 1280],
        # width_list=[768, 512],
        act_func="relu",
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = EfficientViTCls(backbone, head)
    return model


def efficientvit_cls_b1(**kwargs) -> EfficientViTCls:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b1

    backbone = efficientvit_backbone_b1(**kwargs)

    head = ClsHead(
        in_channels=256,
        # width_list=[1536, 1600],
        width_list=[768, 512],
        act_func="relu",
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = EfficientViTCls(backbone, head)
    return model


def efficientvit_cls_b2(**kwargs) -> EfficientViTCls:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b2

    backbone = efficientvit_backbone_b2(**kwargs)

    head = ClsHead(
        in_channels=384,
        width_list=[2304, 2560],
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = EfficientViTCls(backbone, head)
    return model


def efficientvit_cls_b3(**kwargs) -> EfficientViTCls:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b3

    backbone = efficientvit_backbone_b3(**kwargs)

    head = ClsHead(
        in_channels=512,
        width_list=[2304, 2560],
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = EfficientViTCls(backbone, head)
    return model


def efficientvit_cls_l1(**kwargs) -> EfficientViTCls:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_l1

    backbone = efficientvit_backbone_l1(**kwargs)

    head = ClsHead(
        in_channels=512,
        width_list=[3072, 3200],
        act_func="gelu",
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = EfficientViTCls(backbone, head)
    return model


def efficientvit_cls_l2(**kwargs) -> EfficientViTCls:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_l2

    backbone = efficientvit_backbone_l2(**kwargs)

    head = ClsHead(
        in_channels=512,
        width_list=[3072, 3200],
        act_func="gelu",
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = EfficientViTCls(backbone, head)
    return model


def efficientvit_cls_l3(**kwargs) -> EfficientViTCls:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_l3

    backbone = efficientvit_backbone_l3(**kwargs)

    head = ClsHead(
        in_channels=1024,
        width_list=[6144, 6400],
        act_func="gelu",
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = EfficientViTCls(backbone, head)
    return model


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    from DNN_printer import DNN_printer
    from thop import profile, clever_format
    input_shape = (3,256,256)
    input_data = torch.randn(1, input_shape[0], input_shape[1], input_shape[2]).cuda()
    model = efficientvit_cls_b0().cuda().eval()
    # model = efficientvit_cls_b1().cuda().eval()

    # summary(model, input_shape)
    print(model)
    DNN_printer(model, input_shape, batch_size=1, device='cuda')
    MACs, params = profile(model, inputs=(input_data,))
    MACs, params = clever_format([MACs, params], '%.3f')

    print(f"运算量：{MACs}, 参数量：{params}")
