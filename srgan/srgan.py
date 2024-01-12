"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 srgan.py'
"""

import argparse
import os
import numpy as np
import math
import itertools
import sys
from torch import multiprocessing as mp
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader,DistributedSampler
from torch.autograd import Variable
import logging
import coloredlogs
from models import *
from datasets import ImageDataset
from torch import distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm
from utils import device_initializer,load_GPUS
logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO")

os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)
import wandb




def train(rank = None,args = None):
    
    torch.autograd.set_detect_anomaly(True)
    cuda = torch.cuda.is_available()
    if args.distributed and torch.cuda.device_count() > 1 and torch.cuda.is_available():
        distributed = True
        world_size = args.world_size
        # Set address and port
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12347"
        # The total number of processes is equal to the number of graphics cards
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", rank=rank,
                                world_size=world_size)
        # Set device ID
        device = torch.device("cuda", rank)
        # There may be random errors, using this function can reduce random errors in cudnn
        # torch.backends.cudnn.deterministic = True
        # Synchronization during distributed training
        dist.barrier()
        # If the distributed training is not the main GPU, the save model flag is False
        if dist.get_rank() != args.main_gpu:
            save_models = False
        logger.info(msg=f"[{device}]: Successfully Use distributed training.")
    else:
        distributed = False
        # Run device initializer
        device = device_initializer()
        logger.info(msg=f"[{device}]: Successfully Use normal training.")
    if not distributed or rank == 0:
        wandb.init(project="srgan", name="run_epoch_1500")
        config = wandb.config
        config.update(args)  # Log hyperparameters
    if distributed:
        dist.barrier()
    hr_shape = ((args.hr_height//5)*4, (args.hr_height//5)*4)

    # Initialize generator and discriminator
    generator = GeneratorResNet().to(device)
    discriminator_cpu = Discriminator(input_shape=(args.channels, *hr_shape)).to(device)
    feature_extractor = FeatureExtractor().to(device)

    if distributed:
        generator = nn.parallel.DistributedDataParallel(module=generator, device_ids=[device], find_unused_parameters=True)
        discriminator = nn.parallel.DistributedDataParallel(module=discriminator_cpu, device_ids=[device], find_unused_parameters=True)
        feature_extractor = nn.parallel.DistributedDataParallel(module=feature_extractor, device_ids=[device], find_unused_parameters=True)
    # Set feature extractor to inference mode
    feature_extractor.eval()

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_content = torch.nn.L1Loss()

    # if cuda:
    #     generator = generator.cuda()
    #     discriminator = discriminator.cuda()
    #     feature_extractor = feature_extractor.cuda()
    #     criterion_GAN = criterion_GAN.cuda()
    #     criterion_content = criterion_content.cuda()

    if args.epoch != 0:
        # Load pretrained models
        # generator = load_GPUS(generator,"saved_models/generator_499_bestGloss.pth")
        if distributed:
            generator.module.load_state_dict(torch.load("saved_models/generator_1499_bestGloss.pth"))
            discriminator.module.load_state_dict(torch.load("saved_models/discriminator_1499_bestGloss.pth"))
        else:
            generator.load_state_dict(torch.load("saved_models/generator_1499_bestGloss.pth"))
            discriminator.load_state_dict(torch.load("saved_models/discriminator_1499_bestGloss.pth"))
        # discriminator = load_GPUS(discriminator,"saved_models/discriminator_499_bestGloss.pth")

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    if not distributed:
        dataloader = DataLoader(
            ImageDataset("/mnt/data10t/bakuphome20210617/yaoxy/Diffusion-Model/datasets/HuxingDataset", hr_shape=hr_shape),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.n_cpu,
        )
    else:
        sampler = DistributedSampler(ImageDataset("/mnt/data10t/bakuphome20210617/yaoxy/Diffusion-Model/datasets/HuxingDataset", hr_shape=hr_shape))
        dataloader = DataLoader(dataset=ImageDataset("/mnt/data10t/bakuphome20210617/yaoxy/Diffusion-Model/datasets/HuxingDataset", hr_shape=hr_shape), batch_size=args.batch_size,
                                num_workers=args.n_cpu,
                                pin_memory=True, sampler=sampler)
    # ----------
    #  Training
    # ----------
    pbar = tqdm(range(args.epoch, args.n_epochs))
    low_D_loss = 10000000000
    low_G_loss = 10000000000
    for epoch in pbar:
        loss_D_sum = 0
        loss_G_sum = 0
        for i, imgs in enumerate(dataloader):

            # Configure model input
            imgs_lr = Variable(imgs["lr"].type(Tensor)).to(device)
            imgs_hr = Variable(imgs["hr"].type(Tensor)).to(device)
            
            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator_cpu.output_shape))), requires_grad=False).to(device)
            fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator_cpu.output_shape))), requires_grad=False).to(device)

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)
            # print(imgs_lr.shape,gen_hr.shape,valid.shape,*discriminator.output_shape)

            # Adversarial loss
            loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

            # Content loss
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs_hr)
            loss_content = criterion_content(gen_features, real_features.detach())

            # Total loss
            loss_G = loss_content + 1e-3 * loss_GAN

            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss of real and fake images
            loss_real = criterion_GAN(discriminator(imgs_hr), valid)
            loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

            # Total loss
            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            optimizer_D.step()
            loss_D_sum += loss_D.item()
            loss_G_sum += loss_G.item()
            

            batches_done = epoch * len(dataloader) + i
            if batches_done % args.sample_interval == 0:
                # Save image grid with upsampled inputs and SRGAN outputs
                if (distributed and rank == 0) or not distributed:
                    imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=8)
                    gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
                    imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
                    img_grid = torch.cat((imgs_lr, gen_hr), -1)
                    save_image(img_grid, "images/%d.png" % batches_done, normalize=False)

        # --------------
        #  Log Progress
        # --------------

        pbar.set_postfix({"D loss": loss_D_sum, "G loss": loss_G_sum})
        if not distributed or rank==0:
            wandb.log({"loss_D": loss_D_sum, "loss_G": loss_G_sum, "epoch": epoch})
        
        if not distributed or rank == 0:
            if loss_D_sum < low_D_loss:
                # Save model checkpoints
                if distributed:
                    torch.save(generator.module.state_dict(), "saved_models/generator_%d_bestDloss.pth" % epoch)
                    torch.save(discriminator.module.state_dict(), "saved_models/discriminator_%d_bestDloss.pth" % epoch)
                else:
                    torch.save(generator.state_dict(), "saved_models/generator_%d_bestDloss.pth" % epoch)
                    torch.save(discriminator.state_dict(), "saved_models/discriminator_%d_bestDloss.pth" % epoch)
                logger.info(msg=f"[{device}]: Successfully save D best model.")
            if loss_G_sum < low_G_loss:
                # Save model checkpoints
                if distributed:
                    torch.save(generator.module.state_dict(), "saved_models/generator_%d_bestGloss.pth" % epoch)
                    torch.save(discriminator.module.state_dict(), "saved_models/discriminator_%d_bestGloss.pth" % epoch)
                else:
                    torch.save(generator.state_dict(), "saved_models/generator_%d_bestGloss.pth" % epoch)
                    torch.save(discriminator.state_dict(), "saved_models/discriminator_%d_bestGloss.pth" % epoch)
                logger.info(msg=f"[{device}]: Successfully save G best model.")
            if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
                # Save model checkpoints
                if distributed:
                    torch.save(generator.module.state_dict(), "saved_models/generator_%d.pth" % epoch)
                    torch.save(discriminator.module.state_dict(), "saved_models/discriminator_%d.pth" % epoch)
                    logger.info(msg=f"[{device}]: Successfully save model of {epoch}")
                else:
                    torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
                    torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)
                    logger.info(msg=f"[{device}]: Successfully save model of {epoch}")
        if distributed:
            dist.barrier() # wait until all subprocess complete

def main(args):

    if args.distributed:
        os.environ["CUDA_VISIBLE_DEVICES"] = "4,6,7"
        gpus = torch.cuda.device_count()
        mp.spawn(train, args=(args,), nprocs=gpus)
    
    else:
        train(args=args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed",type=bool,default=False)
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
    parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--main_gpu", type=int, default=0)
    args = parser.parse_args()
    
    
    main(args)