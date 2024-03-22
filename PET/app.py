from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import subprocess
import pickle

import sys
import streamlit as st
import torch
from PIL import Image

import argparse
import random
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

import datasets
from datasets import build_dataset
import util.misc as utils
from engine import evaluate
from models import build_model

import cv2
import torchvision.transforms as standard_transforms
from PIL import Image
from tqdm import tqdm
import base64

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def visualization(samples, pred, split_map=None):
    """
    Visualize predictions
    """

    pil_to_tensor = standard_transforms.ToTensor()

    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])

    images = samples
    
    # masks = samples.mask
    for idx in range(images.shape[0]):
        sample = restore_transform(images[idx])
        sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
        sample_vis = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()
        # draw ground-truth points (red)
        size = 2

        # draw predictions (green)
        for p in pred[idx]:
            sample_vis = cv2.circle(sample_vis, (int(p[1]), int(p[0])), size, (0, 255, 0), -1)

        # draw split map
        if split_map is not None:
            imgH, imgW = sample_vis.shape[:2]
            split_map = (split_map * 255).astype(np.uint8)
            split_map = cv2.applyColorMap(split_map, cv2.COLORMAP_JET)
            split_map = cv2.resize(split_map, (imgW, imgH), interpolation=cv2.INTER_NEAREST)
            sample_vis = split_map * 0.9 + sample_vis

        sample_vis = np.clip(sample_vis, 0, 255)
        sample_vis = np.uint8(sample_vis)

    return sample_vis


with open('args.pkl', 'rb') as f:
    args = pickle.load(f)

with open('args2.pkl', 'rb') as f:
    args_2 = pickle.load(f)



app = Flask(__name__, static_url_path='', static_folder='web')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/process-image', methods=['POST'])
def process_image():
    file = request.files['file']
    filename = file.filename
    filepath = os.path.join('uploads', filename)
    file.save(filepath)

    image = Image.open(filepath)
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion = build_model(args)
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # load pretrained model
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        # print(checkpoint['model'])
        model_without_ddp.load_state_dict(checkpoint['model'])
        cur_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0

    # read image
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225]),
    ])
    
    img = transform(image)
    img = torch.Tensor(img).to(device)
    img.unsqueeze_(0)
    img_2 = img.clone()
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 10
    timings = np.zeros((repetitions, 1))

    model.eval()
    outputs = model(img, test=True)
    outputs_points = outputs['pred_points'][0]
    
    img_h,img_w =outputs['img_shape'][0], outputs['img_shape'][1]
    points = [[point[0]*img_h, point[1]*img_w] for point in outputs_points]     # recover to actual points
    split_map = (outputs['split_map_raw'][0].detach().cpu().squeeze(0) > 0.5).float().numpy()
    sample_vis = visualization(img, [points],split_map=split_map)

    with torch.no_grad():
        for rep in tqdm(range(repetitions)):
            starter.record()
            test_stats = model(img, test=True)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
            
    mean_syn = np.sum(timings) / repetitions

    _, buffer = cv2.imencode('.jpg', sample_vis)
    sample_vis_base64 = base64.b64encode(buffer).decode('utf-8')

    del model
    ########### model_2
    model, criterion = build_model(args_2)
    model.to(device)
    model_without_ddp = model

    # load pretrained model
    if args_2.resume:
        if args_2.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args_2.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args_2.resume, map_location='cpu')

        model_without_ddp.load_state_dict(checkpoint['model'])
        cur_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 10
    timings = np.zeros((repetitions, 1))
    model.eval()
    outputs = model(img_2, test=True)

    outputs_points = outputs['pred_points'][0]
    points = [[point[0]*img_h, point[1]*img_w] for point in outputs_points]     # recover to actual points
    split_map = (outputs['split_map_raw'][0].detach().cpu().squeeze(0) > 0.5).float().numpy()
    sample_vis_2 = visualization(img_2, [points],split_map=split_map)

    with torch.no_grad():
        for rep in tqdm(range(repetitions)):
            starter.record()
            test_stats = model(img, test=True)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
            
    mean_syn_2 = np.sum(timings) / repetitions

    _, buffer = cv2.imencode('.jpg', sample_vis_2)
    sample_vis_2_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'mean_syn': round(mean_syn,3), 'image_data': sample_vis_base64, 'mean_syn_2': round(mean_syn_2,3), 'image_data_2': sample_vis_2_base64})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=30118,debug=True)