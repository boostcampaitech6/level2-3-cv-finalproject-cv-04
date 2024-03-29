import re 
import os
import glob
import random
import argparse

import wandb
import cv2
import numpy as np
import torch
import util.misc as utils
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as standard_transforms

from models import build_model




def natural_keys(text):
    """
    alist.sort(key=natural_keys)를 사용하면, 텍스트에 포함된 숫자를 기준으로 정렬할 수 있습니다.
    """
    return [int(c) if c.isdigit() else c for c in re.split("(\d+)", text)]


def get_args_parser():
    parser = argparse.ArgumentParser("Set Point Query Transformer", add_help=False)

    # model parameters
    # - backbone
    parser.add_argument("--backbone", default="vgg16_bn", type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument("--position_embedding", default="sine", type=str, choices=("sine", "learned", "fourier"),
                        help="Type of positional embedding to use on top of the image features")
    # - transformer
    parser.add_argument("--dec_layers", default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument("--dim_feedforward", default=512, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument("--hidden_dim", default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument("--dropout", default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument("--nheads", default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--transformer_method', default="basic", type=str, help="select your method")
    
    # - pet
    parser.add_argument('--pet_method', default="basic", type=str, help="select your method")

    # loss parameters
    # - matcher
    parser.add_argument("--set_cost_class", default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument("--set_cost_point", default=0.05, type=float,
                        help="SmoothL1 point coefficient in the matching cost")
    # - loss coefficients
    parser.add_argument("--ce_loss_coef", default=1.0, type=float)       # classification loss coefficient
    parser.add_argument("--point_loss_coef", default=5.0, type=float)    # regression loss coefficient
    parser.add_argument("--eos_coef", default=0.5, type=float,
                        help="Relative classification weight of the no-object class")   # cross-entropy weights

    # dataset parameters
    parser.add_argument("--dataset_file", default="SHA")
    parser.add_argument("--data_path", default="./data/ShanghaiTech/PartA", type=str)

    # misc parameters
    parser.add_argument("--device", default="cuda",
                        help="device to use for training / testing")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--vis_dir", default="")
    parser.add_argument("--num_workers", default=2, type=int)

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int,
                        help="number of distributed processes")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    
    # measure mode
    parser.add_argument("--repetitions", type=int, default=10)
    
    # wandb
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--wandb_name", type=str, default="default")
    return parser


def main(args):
    # logging with wandb
    if args.use_wandb:
        wandb.init(
            entity="level2_cv4_dc",
            project="final-nota-inference",
            name=args.wandb_name,
            config=args)
    
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion = build_model(args)
    model.to(device)
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("params:", n_parameters / 1e6)

    # load pretrained model
    checkpoint = torch.load(args.resume, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    # get image paths
    img_dir = "./data/ShanghaiTech/part_A/test_data/images"
    img_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
    num_images = len(img_paths)

    # natural_keys 함수를 사용하여 정렬
    img_paths.sort(key=natural_keys)
    
    transform = standard_transforms.Compose(
        [
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    repetitions = args.repetitions
    timings = np.zeros((num_images, repetitions))
    
    img = cv2.imread("./IMG_57.jpg")
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = transform(img)
    img = torch.Tensor(img).to(device)
    img.unsqueeze_(0)
    for _ in range(10):
        _ = model(img, test=True)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    for i, img_path in enumerate(tqdm(img_paths)):
        img = cv2.imread(img_path)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = transform(img)
        img = torch.Tensor(img).to(device)
        img.unsqueeze_(0)
        
        with torch.no_grad():
            for j in range(repetitions):
                starter.record()
                _ = model(img, test=True)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[i, j] = curr_time
        if args.use_wandb:
            wandb.log({"Avg Inference Time per Image": np.mean(timings[i])})

    mean_time = np.mean(timings)
    std_dev = np.std(timings)
    
    if args.use_wandb:
        wandb.log({
            "Avg Inference Time": mean_time,
            "Std Dev": std_dev
            })
        
    print(f"Mean time = {mean_time}ms, std = {std_dev}ms")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("PET evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)