import argparse
import math
import os
import sys

import torch
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
    parser.add_argument("--gpu", type=str, default="7")
    parser.add_argument("--batch_size", help="batch size per gpu", type=int, default=256)
    parser.add_argument("-j", "--workers", help="number of workers", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--crop_ratio", type=float, default=0.95)
    parser.add_argument("--model", type=str, default="efficientvit-b1")
    # parser.add_argument("--weight_url", type=str, default=os.path.join(ROOT_DIR, 'logs/efficientvit_cls/efficientvit_b1/checkpoint/model_best.pt'))
    parser.add_argument("--weight_url", type=str, default=os.path.join(ROOT_DIR, 'logs/efficientvit_cls/efficientvit_b1/checkpoint/checkpoint.pt'))

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

    checkpoint = load_state_dict_from_file(args.weight_url, only_state_dict=False)
    if "ema" in checkpoint and checkpoint["ema"] is not None:
        state_dict = next(iter(checkpoint["ema"].values()))
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model = create_efficientvit_cls_model(args.model, pretrained=False)
    model.load_state_dict(state_dict)
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
