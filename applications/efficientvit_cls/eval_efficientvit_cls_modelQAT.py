import argparse
import math
import os
import sys

import torch.utils.data
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

from efficientvit.apps.utils import AverageMeter
from efficientvit.cls_model_zoo import create_efficientvit_cls_model
from efficientvit.models.utils import load_state_dict_from_file

from mqbench_export.prepare_by_platform import prepare_by_platform
from applications.quant_config import BackendMap, prepare_custom_config_dict



def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list[torch.Tensor]:
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/srv/datasets/ImageNet100/val")
    parser.add_argument("--gpu", type=str, default="5")
    parser.add_argument("--batch_size", help="batch size per gpu", type=int, default=256)
    parser.add_argument("-j", "--workers", help="number of workers", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--crop_ratio", type=float, default=0.95)
    parser.add_argument("--model", type=str, default="efficientvit-b1")
    # parser.add_argument("--weight_url", type=str, default=os.path.join(ROOT_DIR, 'logs/efficientvit_cls/efficientvit_b1_QAT/checkpoint/model_best.pt'))
    parser.add_argument("--weight_url", type=str, default=os.path.join(ROOT_DIR, 'logs/efficientvit_cls/efficientvit_b1_QAT/checkpoint/checkpoint.pt'))

    quant_backend = BackendMap['tensorrt'] #['tensorrt', 'nnie', 'ppl', 'snpe', 'vitis', 'tengine_u8']
    args = parser.parse_args()
    if args.gpu == "all":
        device_list = range(torch.cuda.device_count())
        args.gpu = ",".join(str(_) for _ in device_list)
    else:
        device_list = [int(_) for _ in args.gpu.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.batch_size = args.batch_size * max(len(device_list), 1)

    args.path = os.path.expanduser(args.path)
    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            args.path,
            transforms.Compose(
                [
                    transforms.Resize(
                        int(math.ceil(args.image_size / args.crop_ratio)), interpolation=InterpolationMode.BICUBIC
                    ),
                    transforms.CenterCrop(args.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    model = create_efficientvit_cls_model(args.model, pretrained=False, dropout=0.05)
    model = prepare_by_platform(model, quant_backend, prepare_custom_config_dict)
    weight = load_state_dict_from_file(args.weight_url)
    model.load_state_dict(weight)

    # from DNN_printer import DNN_printer
    # from thop import profile, clever_format
    # input_shape = (3,256,256)
    # device  = "cuda" if torch.cuda.is_available() else "cpu"
    # model = model.to(device)
    # print(model)
    # DNN_printer(model, input_shape, batch_size=1, device='cuda')
    # input_data = torch.randn(1, input_shape[0], input_shape[1], input_shape[2]).to(device)
    # MACs, params = profile(model, inputs=(input_data,))
    # MACs, params = clever_format([MACs, params], '%.3f')
    # print(f"运算量：{MACs}, 参数量：{params}")

    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    top1 = AverageMeter(is_distributed=False)
    top5 = AverageMeter(is_distributed=False)
    with torch.inference_mode():
        with tqdm(total=len(data_loader), desc=f"Eval {args.model} on ImageNet") as t:
            for images, labels in data_loader:
                images, labels = images.cuda(), labels.cuda()
                # compute output
                output = model(images)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))

                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))
                t.set_postfix(
                    {
                        "top1": top1.avg,
                        "top5": top5.avg,
                        "resolution": images.shape[-1],
                    }
                )
                t.update(1)

    print(f"Top1 Acc={top1.avg:.3f}, Top5 Acc={top5.avg:.3f}")


if __name__ == "__main__":
    main()
