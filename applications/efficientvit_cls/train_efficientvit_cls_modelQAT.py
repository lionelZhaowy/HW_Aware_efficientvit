import argparse
import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

from efficientvit.apps import setup
from efficientvit.apps.utils import dump_config, is_master, parse_unknown_args
from efficientvit.cls_model_zoo import create_efficientvit_cls_model
from efficientvit.clscore.data_provider import ImageNetDataProvider
from efficientvit.clscore.trainer import ClsRunConfig, ClsTrainer
# from efficientvit.models.nn.drop import apply_drop_func
from efficientvit.models.utils import load_state_dict_from_file

from mqbench_export.prepare_by_platform import prepare_by_platform, BackendType
from mqbench_export.utils.state import enable_calibration, enable_quantization
from applications.quant_config import BackendMap, prepare_custom_config_dict

parser = argparse.ArgumentParser()
parser.add_argument("config", metavar="FILE", help="config file", 
                    default="applications/efficientvit_cls/configs/imagenet/efficientvit_b1_QAT.yaml")
parser.add_argument("--path", type=str, metavar="DIR", help="run directory",
                    default=os.path.join(ROOT_DIR, "logs/efficientvit_cls/efficientvit_b1_QAT"))
parser.add_argument("--gpu", type=str, default=None)  # used in single machine experiments
parser.add_argument("--manual_seed", type=int, default=0)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--amp", type=str, choices=["fp32", "fp16", "bf16"], default="fp32")

# initialization
parser.add_argument("--rand_init", type=str, default="trunc_normal@0.02")
parser.add_argument("--last_gamma", type=float, default=0)
parser.add_argument("--pretrainedQAT", type=str, default='')

parser.add_argument("--auto_restart_thresh", type=float, default=8.0)
parser.add_argument("--save_freq", type=int, default=1)
parser.add_argument("--weight_url", type=str, default=os.path.join(ROOT_DIR, 'logs/efficientvit_cls/efficientvit_b1/checkpoint/model_best.pt'))


def calibrate_with_train_data(data_provider, model, cali_images=16000, cali_batch_size=128):
    """Calibrate using training subset covering MULTIPLE resolutions.

    The original calibrate() only used 256 images at a single validation
    resolution, which is insufficient for EMAMinMaxObserver to capture
    the full activation range across all training resolutions.

    This function:
    1. Uses training data (not validation) for calibration
    2. Calibrates at each training resolution
    3. Uses enough images (default 4000) for proper observer convergence
    4. Runs multiple forward passes to let EMA observer converge
    """
    model.eval()
    print("Start multi-resolution calibration ...")
    training_resolutions = data_provider.image_size  # e.g., [(160,160), (192,192), (224,224), (256,256)]
    if not isinstance(training_resolutions, list):
        training_resolutions = [training_resolutions]

    total_images = 0
    for res in training_resolutions:
        print(f"  Calibrating at resolution {res} ...")
        data_provider.assign_active_image_size(res)
        cali_loader = data_provider.build_sub_train_loader(
            n_samples=cali_images // len(training_resolutions),
            batch_size=cali_batch_size,
        )
        with torch.no_grad():
            for i, images in enumerate(cali_loader):
                # build_sub_train_loader returns raw image tensors (no labels)
                if isinstance(images, (list, tuple)):
                    images = images[0]
                images = images.cuda(non_blocking=True)
                _ = model(images)
        total_images += len(cali_loader) * cali_batch_size
        print(f"    Done ({len(cali_loader)} batches).")

    print(f"End multi-resolution calibration (total ~{total_images} images across {len(training_resolutions)} resolutions).")
    return


def fix_input_fakequantize_scale(model, input_scale=1.0 / 128.0):
    """Match the NPU UINT8-to-INT8 (value - 128) input conversion."""
    input_node = next((node for node in model.graph.nodes if node.op == "placeholder"), None)
    if input_node is None:
        raise RuntimeError("Cannot find the model input node.")

    fq_node = next(
        (
            node
            for node in model.graph.nodes
            if node.op == "call_module"
            and "post_act_fake_quantizer" in str(node.target)
            and node.args
            and node.args[0] == input_node
        ),
        None,
    )
    if fq_node is None:
        raise RuntimeError("Cannot find the input FakeQuantize module.")

    fq_module = model.get_submodule(fq_node.target)
    fq_module.scale.data.fill_(input_scale)
    fq_module.scale.requires_grad = False


def main():
    # parse args
    args, opt = parser.parse_known_args()
    opt = parse_unknown_args(opt)

    # setup quantization backend
    quant_backend = BackendMap['tensorrt'] #['tensorrt', 'nnie', 'ppl', 'snpe', 'vitis', 'tengine_u8']

    # setup gpu and distributed training
    setup.setup_dist_env(args.gpu)

    # setup path, update args, and save args to path
    os.makedirs(args.path, exist_ok=True)
    dump_config(args.__dict__, os.path.join(args.path, "args.yaml"))

    # setup random seed
    setup.setup_seed(args.manual_seed, args.resume)

    # setup exp config
    config = setup.setup_exp_config(args.config, recursive=True, opt_args=opt)

    # save exp config
    setup.save_exp_config(config, args.path)

    # setup data provider
    data_provider = setup.setup_data_provider(config, [ImageNetDataProvider], is_distributed=True)

    # setup run config
    run_config = setup.setup_run_config(config, ClsRunConfig)

    # setup model
    model = create_efficientvit_cls_model(
        config["net_config"]["name"],
        pretrained=False,  # 不让 create 函数自动加载原始权重
        dropout=config["net_config"]["dropout"]
    )
    # apply_drop_func(model.backbone.stages, config["backbone_drop"])

    # 手动加载 checkpoint，提取 EMA 权重（而非原始 state_dict）
    weight_url = os.path.realpath(os.path.expanduser(args.weight_url))
    checkpoint = torch.load(weight_url, map_location="cpu", weights_only=True)

    if "ema" in checkpoint and checkpoint["ema"] is not None:
        ema_state_dict = checkpoint["ema"]
        # EMA state_dict 格式: {decay值: 模型state_dict}
        if isinstance(ema_state_dict, dict) and len(ema_state_dict) > 0:
            decay = list(ema_state_dict.keys())[0]
            model.load_state_dict(ema_state_dict[decay])
            print(f"已加载 EMA 权重 (decay={decay:.4f}) 用于 QAT 初始化")
        else:
            model.load_state_dict(checkpoint["state_dict"])
            print("警告: EMA 键存在但为空，回退到原始权重")
    else:
        model.load_state_dict(checkpoint["state_dict"])
        print("警告: checkpoint 中无 EMA，加载原始权重")

    model = prepare_by_platform(model, quant_backend, prepare_custom_config_dict)
    if args.pretrainedQAT:
        if not os.path.isfile(args.pretrainedQAT):
            raise FileNotFoundError(args.pretrainedQAT)
        weight = load_state_dict_from_file(args.pretrainedQAT)
        model.load_state_dict(weight)

    # setup trainer
    trainer = ClsTrainer(
        path=args.path,
        model=model,
        data_provider=data_provider,
        auto_restart_thresh=args.auto_restart_thresh,
    )
    # # initialization
    # setup.init_model(
    #     trainer.network,
    #     rand_init=args.rand_init,
    #     last_gamma=args.last_gamma,
    # )

    # prep for training
    trainer.prep_for_training(run_config, config["ema_decay"], args.amp)

    # resume
    if args.resume:
        trainer.load_model()
        trainer.data_provider = setup.setup_data_provider(config, [ImageNetDataProvider], is_distributed=True)
    else:
        trainer.sync_model()

    if not args.resume and not args.pretrainedQAT:
        enable_calibration(trainer.network)
        trainer.data_provider.set_epoch(0)
        calibrate_with_train_data(trainer.data_provider, trainer.network, cali_images=4000, cali_batch_size=128)
        enable_quantization(trainer.network)
        fix_input_fakequantize_scale(trainer.network)

    # ================================================================
    # 校准后验证 (Pre-QAT calibration validation)
    # ================================================================
    if is_master():
        print("=" * 60)
        print("校准后验证 — 评估量化模型在QAT训练前的精度")
        print("=" * 60)
    calib_val_info = trainer.multires_validate(is_test=True, epoch=-1)
    if is_master():
        for res_key, info in calib_val_info.items():
            print(f"  [{res_key}] Top1={info['val_top1']:.2f}%", end="")
            if "val_top5" in info:
                print(f", Top5={info['val_top5']:.2f}%", end="")
            print(f", Loss={info['val_loss']:.4f}")
        print("=" * 60)

    if not args.resume and not args.pretrainedQAT:
        trainer.best_val = sum(info["val_top1"] for info in calib_val_info.values()) / len(calib_val_info)
        trainer.save_model(only_state_dict=False, epoch=-1, model_name="model_best.pt")

    # launch training
    trainer.train(save_freq=args.save_freq)


if __name__ == "__main__":
    main()
