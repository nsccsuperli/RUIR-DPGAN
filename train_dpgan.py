"""
 > Maintainer: https://github.com/nsccsuperli/DPGAN
"""

# py libs licht
import os
import sys
import yaml
import argparse
import numpy as np
from PIL import Image
# pytorch libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
# local libs
from nets.commons import Weights_Normal, VGG19_EdgeLoss
from nets.dpgan import GeneratorDPGAN, DiscriminatorDPGAN
from utils.data_utils import GetTrainingPairs, GetValImage
from torchvision.transforms import *
## get configs and training options
parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", type=str, default="configs/train_euvp.yaml")
parser.add_argument("--epoch", type=int, default=0, help="which epoch to start from")
parser.add_argument("--num_epochs", type=int, default=10, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0003, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of 1st order momentum")
parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of 2nd order momentum")
args = parser.parse_args()

## training params
epoch = args.epoch
num_epochs = args.num_epochs
batch_size = args.batch_size
lr_rate, lr_b1, lr_b2 = args.lr, args.b1, args.b2
# load the data config file
with open(args.cfg_file) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
# get info from config file
dataset_name = cfg["dataset_name"]
dataset_path = cfg["dataset_path"]
channels = cfg["chans"]
img_width = cfg["im_width"]
img_height = cfg["im_height"]
val_interval = cfg["val_interval"]
ckpt_interval = cfg["ckpt_interval"]


## create dir for model and validation data
samples_dir = os.path.join("samples/DPGAN/", dataset_name)
checkpoint_dir = os.path.join("checkpoints/DPGAN/", dataset_name)
os.makedirs(samples_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

""" RUIR-DPGAN specifics: loss functions and patch-size
-----------------------------------------------------"""

Adv_cGAN = torch.nn.MSELoss()
L1_G  = torch.nn.L1Loss() # similarity loss (l1)
MSELoss = torch.nn.MSELoss()
Edge_vgg = VGG19_EdgeLoss() # Edge loss
lambda_1,lambda_mse,lambda_con_edg = 6,6, 12 # weights coefficient
patch = (1, img_height//16, img_width//16) # 16x16 for 256x256

# Initialize generator and discriminator
generator = GeneratorDPGAN()
discriminator = DiscriminatorDPGAN()

# see if cuda is available
if torch.cuda.is_available():
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    Adv_cGAN.cuda()
    L1_G = L1_G.cuda()
    Edge_vgg = Edge_vgg.cuda()
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

# Initialize weights or load pretrained models
if args.epoch == 0:
    generator.apply(Weights_Normal)
    discriminator.apply(Weights_Normal)
else:
    generator.load_state_dict(torch.load("checkpoints/RUIR-DPGAN/%s/generator_%d.pth" % (dataset_name, args.epoch)))
    discriminator.load_state_dict(torch.load("checkpoints/RUIR-DPGAN/%s/discriminator_%d.pth" % (dataset_name, epoch)))
    print ("Loaded model from epoch %d" %(epoch))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_rate, betas=(lr_b1, lr_b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_rate, betas=(lr_b1, lr_b2))


## Data pipeline
transforms_ = [
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    GetTrainingPairs(dataset_path, dataset_name, transforms_=transforms_),
    batch_size = batch_size,
    shuffle = True,
    num_workers = 8,
)

val_dataloader = DataLoader(
    GetValImage(dataset_path, dataset_name, transforms_=transforms_, sub_dir='validation'),
    batch_size=4,
    shuffle=True,
    num_workers=1,
)


def tensor_to_PIL(variable):  # Variable To Image
    image = variable.cpu().clone()
    image = image.data  # Variable To tensor
    for i in range(len(image)):
        save_file = "./record/process_img/up4/" + str(i) + ".jpg"
        temp_image = image[i].clone()
        temp_image = temp_image.unsqueeze(0)  # unsqueeze
        temp_image = transforms.ToPILImage()(temp_image.float()).convert("L")  # To PILImage RGB 3 channel
        temp_image.save(save_file, quality=95)

## start Training
for epoch in range(epoch, num_epochs):
    for i, batch in enumerate(dataloader):
        # Model inputs
        imgs_distorted = Variable(batch["A"].type(Tensor))
        # tensor_to_PIL(imgs_distorted)

        imgs_good_gt = Variable(batch["B"].type(Tensor))
        img_sample = torch.cat((imgs_distorted.data, imgs_good_gt.data), -2)

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_distorted.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_distorted.size(0), *patch))), requires_grad=False)

        ## Train Discriminator
        optimizer_D.zero_grad()
        # Generate simulation images based on existing images
        imgs_fake = generator(imgs_distorted)
        # Send the real image and the original image to the discriminant network, and output the verification result
        pred_real = discriminator(imgs_good_gt, imgs_distorted)
        img_sample_pred_real = torch.cat((imgs_good_gt.data, imgs_distorted.data), -2)
        save_image(img_sample_pred_real.data, "samples/RUIR-DPGAN/%s/%s.png" % (dataset_name, "img_sample_pred_real"),
                   nrow=5, normalize=True)
        # Calculate the loss value
        loss_real = Adv_cGAN(pred_real, valid)
        # Send real images and simulated images to the discriminant network
        pred_fake = discriminator(imgs_fake, imgs_distorted)
        # print(str(pred_fake))
        img_sample_pred_fake = torch.cat((imgs_fake.data, imgs_distorted.data), -2)
        save_image(img_sample_pred_fake.data, "samples/RUIR-DPGAN/%s/%s.png" % (dataset_name, "img_sample_pred_fake"), nrow=5, normalize=True)
        loss_fake = Adv_cGAN(pred_fake, fake)
        save_image(imgs_fake.data, "samples/RUIR-DPGAN/%s/%s.png" % (dataset_name, "imgs_fake"), nrow=5, normalize=True)
        # Total loss: real + fake (standard PatchGAN)
        loss_D = 0.5 * (loss_real + loss_fake) * 10.0 # 10x scaled for stability
        loss_D.backward()
        optimizer_D.step()

        ## Train Generator
        optimizer_G.zero_grad()
        imgs_fake = generator(imgs_distorted)
        pred_fake = discriminator(imgs_fake, imgs_distorted)
        loss_GAN =  Adv_cGAN(pred_fake, valid) # GAN loss
        loss_1 = L1_G(imgs_fake, imgs_good_gt) # similarity loss
        loss_m = MSELoss(imgs_fake,imgs_good_gt)
        loss_edg = Edge_vgg(imgs_fake, imgs_good_gt)# Edge loss

        loss_G = loss_GAN + lambda_1 * loss_1 +lambda_con_edg *loss_edg +loss_m*lambda_mse
        loss_G.backward()
        optimizer_G.step()
        best_loss = loss_G
        sys.stdout.write("\r[Epoch %d/%d: batch %d/%d] [DLoss: %.3f, GLoss: %.3f, AdvLoss: %.3f, EdgeLoss: %.3f,MseLoss: %.3f]"
                         % (
                             epoch, num_epochs, i, len(dataloader),
                             loss_D.item(), loss_G.item(), loss_GAN.item(),loss_edg.item(),loss_m.item()
                         )
                         )
        ## Print log
        if not i%50:
            sys.stdout.write("\r[Epoch %d/%d: batch %d/%d] [DLoss: %.3f, GLoss: %.3f, AdvLoss: %.3f]"
                              %(
                                epoch, num_epochs, i, len(dataloader),
                                loss_D.item(), loss_G.item(), loss_GAN.item(),
                               )
            )
        ## If at sample interval save image
        batches_done = epoch * len(dataloader) + i
        if batches_done % val_interval == 0:
            imgs = next(iter(val_dataloader))
            imgs_val = Variable(imgs["val"].type(Tensor))
            imgs_gen = generator(imgs_val)
            img_sample = torch.cat((imgs_val.data, imgs_gen.data), -2)
            save_image(img_sample, "samples/RUIR-DPGAN/%s/%s.png" % (dataset_name, batches_done), nrow=5, normalize=True)

    ## Save model checkpoints
    if (epoch % ckpt_interval == 0):

        torch.save(generator.state_dict(), "checkpoints/RUIR-DPGAN/%s/generator_%d.pth" % (dataset_name, epoch))
        torch.save(discriminator.state_dict(), "checkpoints/RUIR-DPGAN/%s/discriminator_%d.pth" % (dataset_name, epoch))


