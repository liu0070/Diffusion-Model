import sys,os
sys.path.append(os.path.dirname(sys.path[0]))
from .models import *
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image, make_grid
from .utils import load_GPUS

def erode(image,kernel_size):
    image_np = image.detach().cpu().numpy()
    for c in range(image_np.shape[1]):
        for h in range(image_np.shape[2]):
            for w in range(image_np.shape[3]):
                h_s = max(h-kernel_size[0]//2,0)
                h_e = min(h+kernel_size[0]//2+1,image_np.shape[2])
                w_s = max(w-kernel_size[1]//2,0)
                w_e = min(w+kernel_size[1]//2+1,image_np.shape[3])
                # print(h_s,h_e,w_s,w_e)
                window = image_np[0,h_s:h_e,w_s:w_e,c]
                if window.size > 0:
                    image_np[0,c,h,w] = np.max(window)
    
    return torch.from_numpy(image_np).to(image.device).type(image.dtype)
                
def srgan_generate(images):
    
    # Normalization parameters for pre-trained PyTorch models
    mean = np.array([0.92284108,0.90716445,0.86908176])
    std = np.array([0.20057224,0.20696244,0.24172288])
    cuda = torch.cuda.is_available()

    generator = GeneratorResNet().cuda()
    generator.eval()

    generator.load_state_dict(torch.load("/mnt/data10t/bakuphome20210617/yaoxy/Diffusion-Model/srgan/saved_models/generator_2837_bestGloss.pth"))

    image_transform = transforms.Compose(
                [
                    transforms.Normalize(mean, std),
                ]
            )


    imgs_lr = image_transform(images.type(torch.float32))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imgs_lr = imgs_lr.to(device)
    # imgs_lr = erode(imgs_lr,[10,20])

    imgs_hr = generator(imgs_lr)
    return imgs_hr.detach()