#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import random
import warnings

from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.quantization.utils import weight_dtype

from yolox.core import Trainer, launch
from yolox.exp import get_exp

# stephen add 
import sys
from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--local_rank", default=0, type=int, help="local rank for dist training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default='/opt/tiger/minist/YOLOX/exps/example/yolox_voc/yolox_voc_s.py',
        type=str,
        help="plz input your expriment description file",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("-c", "--ckpt", default='/opt/tiger/minist/YOLOX/weight/yolox_s.pth', type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main(exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    cudnn.benchmark = True

    trainer = Trainer(exp, args)
    trainer.train()


if __name__ == "__main__":

    backbone = YOLOPAFPN(in_channels=[128, 256, 2048])
    
    head = YOLOXHead(num_classes = 6, in_channels=[128, 256, 2048])
    model = YOLOX(backbone= backbone, head=head)
    # print(model)
    weight_path = '/opt/tiger/minist/YOLOX/weight/best_top1.pth'
    # pretrain_dict = torch.load(weight_path)
    # model.load_state_dict(pretrain_dict, strict=False)
    model.eval()

    # for k, v in model.named_modules() :
    #     print(k)
    #     print(v)
    #     print(''.center(30, '-'))
    
    x = torch.ones((4, 3, 416, 416))
    y = model(x)
    print(y.shape)
    sys.exit()

    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()
    print('start trainer')
    trainer = Trainer(exp, args)
    print('struct model ing...')
    # print(trainer.exp)
    model = exp.get_model()
    print('model info:')
    print(model)
    x = torch.ones((4, 3, 416, 416))
    y = model(x)
    print(y.shape)


    Command Line Args: 
    Namespace(config_file='configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_1shot.yaml', dist_url='tcp://127.0.0.1:50152', 
    end_iter=-1, eval_all=False, eval_during_train=False, eval_iter=-1, eval_only=False, machine_rank=0, num_gpus=8, num_machines=1, 
    opts=['MODEL.WEIGHTS', 'checkpoints/faster_rcnn_R_101_FPN_all1/model_reset_surgery.pth'], resume=False, start_iter=-1)

    