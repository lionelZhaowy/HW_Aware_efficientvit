import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

from efficientvit.apps.utils import AverageMeter


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
    parser.add_argument("--batch_size", help="batch size per gpu", type=int, default=256)
    parser.add_argument("-j", "--workers", help="number of workers", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--crop_ratio", type=float, default=0.95)
    parser.add_argument("--onnx_path", type=str, default=os.path.join(ROOT_DIR, "onnx/EfficientViT_b1.onnx"))
    parser.add_argument("--gpu", type=str, default="5")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

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

    # Load ONNX model
    onnx_path = os.path.realpath(os.path.expanduser(args.onnx_path))
    print(f"Loading ONNX model from: {onnx_path}")
    sess = ort.InferenceSession(
        onnx_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        provider_options=[{"device_id": 0}, {}],
    )
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print(f"ONNX model input: {input_name}, output: {output_name}")
    print(f"Using GPU: {args.gpu}, Providers: {sess.get_providers()}")

    top1 = AverageMeter(is_distributed=False)
    top5 = AverageMeter(is_distributed=False)
    with tqdm(total=len(data_loader), desc=f"Eval ONNX model on ImageNet") as t:
        for images, labels in data_loader:
            # ONNX model was exported with fixed batch_size=1,
            # so we must feed one sample at a time
            batch_outputs = []
            for i in range(images.size(0)):
                img_np = images[i].unsqueeze(0).numpy().astype(np.float32)
                out_np = sess.run([output_name], {input_name: img_np})[0]
                batch_outputs.append(torch.from_numpy(out_np))

            output = torch.cat(batch_outputs, dim=0)

            # Measure accuracy
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
