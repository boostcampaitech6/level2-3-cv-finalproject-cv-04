from models import build_model
import argparse
import cv2
from PIL import Image
import torchvision.transforms as standard_transforms
import util.misc as utils
import numpy as np
from tqdm import tqdm
import random

import torch


def get_args_parser():
    parser = argparse.ArgumentParser('Set Point Query Transformer', add_help=False)

    # model parameters
    # - backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned', 'fourier'),
                        help="Type of positional embedding to use on top of the image features")
    # - transformer
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=512, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--transformer_method', default="basic", type=str, help="select your method")
    
    # - pet
    parser.add_argument('--pet_method', default="basic", type=str, help="select your method")
    
    # loss parameters
    # - matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="SmoothL1 point coefficient in the matching cost")
    # - loss coefficients
    parser.add_argument('--ce_loss_coef', default=1.0, type=float)       # classification loss coefficient
    parser.add_argument('--point_loss_coef', default=5.0, type=float)    # regression loss coefficient
    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")   # cross-entropy weights

    # dataset parameters
    parser.add_argument('--dataset_file', default="SHA")
    parser.add_argument('--data_path', default="./data/ShanghaiTech/PartA", type=str)

    # misc parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--vis_dir', default="")
    parser.add_argument('--num_workers', default=2, type=int)
    
    # measure mode
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--measure_mode", default="total",
                        help="block: block-wise inference time, layer: layer-wise inference time.")
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser('PET evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    model, criterion = build_model(args)
    model.to(device)
    
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    
    # read image
    img_path = './IMG_57.jpg'
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
    ])
    
    img = transform(img)
    img = torch.Tensor(img).to(device)
    img.unsqueeze_(0)
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = args.repetitions

    for _ in range(10):
        _ = model(img, test=True)
        
    if args.measure_mode == "total":
        timings = np.zeros((repetitions, 1))

        with torch.no_grad():
            for rep in tqdm(range(repetitions)):
                starter.record()
                _ = model(img, test=True)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time

        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)

        print(f"Mean time = {mean_syn}ms, std = {std_syn}ms")
    elif args.measure_mode == "block":
        module_timings = {}
        
        # 모델의 각 모듈에 대한 타이머 설정
        for name, module in model.named_children():
            module_timings[name] = np.zeros((repetitions, 1))

        with torch.no_grad():
            for rep in tqdm(range(repetitions)):
                for name, module in model.named_children():
                    
                    def module_operation_start_time_hook(module, input):
                        starter.record()
                        
                    def module_operation_end_time_hook(module, input, output):
                        ender.record()
                        torch.cuda.synchronize()
                        curr_time = starter.elapsed_time(ender)
                        module_timings[name][rep] += curr_time

                    start = module.register_forward_pre_hook(module_operation_start_time_hook)
                    end = module.register_forward_hook(module_operation_end_time_hook)
                    _ = model(img, test=True)  # 모델 실행
                    start.remove()
                    end.remove()

        # 각 모듈별 평균 추론 시간 및 표준 편차 계산 및 출력
        for name, timings in module_timings.items():
            mean_syn = np.mean(timings)
            std_syn = np.std(timings)
            print(f"Module: {name}, Mean time = {mean_syn}ms, std = {std_syn}ms")
    elif args.measure_mode == "layer":
        module_timings = {}
        
        # 모델의 각 모듈에 대한 타이머 설정
        for name, module in model.named_modules():
            module_timings[name] = np.zeros((repetitions, 1))

        with torch.no_grad():
            for rep in tqdm(range(repetitions)):
                for name, module in model.named_modules():
                    
                    def module_operation_start_time_hook(module, input):
                        starter.record()
                        
                    def module_operation_end_time_hook(module, input, output):
                        ender.record()
                        torch.cuda.synchronize()
                        curr_time = starter.elapsed_time(ender)
                        module_timings[name][rep] += curr_time

                    start = module.register_forward_pre_hook(module_operation_start_time_hook)
                    end = module.register_forward_hook(module_operation_end_time_hook)
                    _ = model(img, test=True)  # 모델 실행
                    start.remove()
                    end.remove()

        # 각 모듈별 평균 추론 시간 및 표준 편차 계산 및 출력
        for name, timings in module_timings.items():
            mean_syn = np.mean(timings)
            std_syn = np.std(timings)
            print(f"Module: {name}, Mean time = {mean_syn}ms, std = {std_syn}ms")