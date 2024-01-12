#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys


import logging

import coloredlogs

import torch
import torchvision

sys.path.append(os.path.dirname(sys.path[0]))
from srgan.generate import srgan_generate
from model.ddpm import Diffusion as DDPMDiffusion
from model.ddim import Diffusion as DDIMDiffusion
from model.network import UNet
from utils.utils import save_images
from utils.initializer import device_initializer, load_model_weight_initializer

logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")


import gradio as gr


        
def generate(sample="ddpm",image_size=72,num_images=1,class_name=-1,cfg_scale=0):
    """
    start generating
    :param sample: sample type optional ddpm,ddim
    :param image_size:the size for generate image
    :param num_images: the num of generate image
    :param class_name: [optional] the class name of target generate image
    :return: None
    """
    houseTypeDict = {
        "2室1厅":0,
        "2室2厅":1,
        "3室1厅":2,
        "3室2厅":3,
        "4室2厅":4,
    }
    logger.info(msg="start generation.")
    logger.info(msg=f"Input params:{sample,image_size,num_images,class_name}")
    device = device_initializer()
    if sample == "ddim":
        weight_path = "/mnt/data10t/bakuphome20210617/yaoxy/Diffusion-Model/resultsddim/dfddim/ema_model_last.pt"
        diffusion = DDIMDiffusion(img_size=image_size,device=device)
    else:
        weight_path = "/mnt/data10t/bakuphome20210617/yaoxy/Diffusion-Model/results/df/ema_model_148.pt"
        diffusion = DDPMDiffusion(img_size=image_size,device=device)
    num_classes = 5
    model = UNet(num_classes=num_classes,device=device,image_size=image_size,act="gelu").to(device)
    load_model_weight_initializer(model=model, weight_path=weight_path, device=device, is_train=False)
    logger.info(msg="successfully load model")
    if class_name == -1:
        y = torch.arange(num_classes).long().to(device)
        num_images = num_classes
    else:
        class_name = houseTypeDict[class_name]
        y = torch.Tensor([class_name] * num_images).long().to(device)
    x = diffusion.sample(model=model, n=num_images, labels=y, cfg_scale=cfg_scale)
    x_srgan = srgan_generate(x)
    grid = torchvision.utils.make_grid(tensor=x)
    grid_srgan = torchvision.utils.make_grid(tensor=x_srgan)
    image_array = grid.permute(1, 2, 0).to("cpu").numpy()
    image_array_srgan = grid_srgan.permute(1,2,0).to("cpu").numpy()
    return image_array,image_array_srgan

gr.Interface(
    generate,
    title="Diffusion model for house type generate",
    inputs=[
        gr.Radio(["ddpm","ddim"],value="ddpm",label="sample method"),
        gr.Number(minimum=0, maximum=1024, label="image size",step=8),
        gr.Number(minimum=1, maximum=5, label="num images"),
        gr.Radio(["2室1厅", "2室2厅", "3室1厅","3室2厅","4室2厅"],value = -1,label="House Type", info="What type of house that you want generate!"),
        gr.Slider(0, 10, value=0, label="CFG Scale", info="classifier-free guidance插值权重"),
    ],
    outputs=[gr.Image(height=250,width=250),gr.Image(height=250,width=250)],
).launch()