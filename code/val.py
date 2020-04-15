#-*- coding:utf-8 -*-
'''
[AI502] Deep Learning Assignment
"Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" Implementation
20193640 Jungwon Choi
'''
import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import os

import cv2
import torchvision.transforms as transforms
import torchvision.transforms.functional as tvF
import torchvision.utils as utils
from util import FIGURE_PATH

#===============================================================================
''' Heatmap colorization '''
def heatmap_colorization(gray_heatmap):
    # tensor -> numpy
    gray_heatmap = np.transpose(gray_heatmap.numpy()*255, (1, 2, 0))
    # Colorization
    color_heatmap = cv2.applyColorMap(gray_heatmap.astype('uint8'),
                                    cv2.COLORMAP_JET)
    # BGR -> RGB
    color_heatmap = color_heatmap[..., ::-1].copy()
    # numpy -> tensor
    color_heatmap = torch.from_numpy(np.transpose(color_heatmap, (2, 0, 1)))
    # Normalize the tensor
    transform = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((.5, .5, .5), (.5, .5, .5))])
    color_heatmap = transform(tvF.to_pil_image(color_heatmap))
    return color_heatmap

#===============================================================================
''' Validate sequence '''
def val(val_dataloader, A_generator, B_generator,
        A_discriminator, B_discriminator, epoch, FILE_NAME_FORMAT, attention):
    A_generator.eval()
    B_generator.eval()
    device = next(A_generator.parameters()).device.index
    total_iter = len(val_dataloader)

    # Check the directory of the file path
    SAVE_IMG_PATH = os.path.join(FIGURE_PATH, FILE_NAME_FORMAT, str(epoch))
    if not os.path.exists(SAVE_IMG_PATH):
        os.makedirs(SAVE_IMG_PATH)

    with torch.no_grad():
        for i, (A_image, B_image) in enumerate(val_dataloader):
            real_A_image = A_image.cuda(device)
            real_B_image = B_image.cuda(device)
            #===================================================================
            # Generate each fake image
            if attention:
                real_attn_A = B_discriminator.forward_attantionmap(real_A_image)
                fake_B_image = A_generator(real_attn_A*real_A_image)
                real_attn_B = A_discriminator.forward_attantionmap(real_B_image)
                fake_A_image = B_generator(real_attn_B*real_B_image)
                fake_attn_B = A_discriminator.forward_attantionmap(fake_B_image)
                rect_A_image = B_generator(fake_attn_B*fake_B_image)
                fake_attn_A = B_discriminator.forward_attantionmap(fake_A_image)
                rect_B_image = B_generator(fake_attn_A*fake_A_image)
            else:
                fake_B_image = A_generator(real_A_image)
                fake_A_image = B_generator(real_B_image)
                rect_A_image = B_generator(fake_B_image)
                rect_B_image = A_generator(fake_A_image)

            # Save the real & fake images
            real_A_image = real_A_image.detach().cpu().squeeze(0)
            real_B_image = real_B_image.detach().cpu().squeeze(0)
            fake_A_image = fake_A_image.detach().cpu().squeeze(0)
            fake_B_image = fake_B_image.detach().cpu().squeeze(0)
            rect_A_image = rect_A_image.detach().cpu().squeeze(0)
            rect_B_image = rect_B_image.detach().cpu().squeeze(0)
            if attention:
                real_attn_A = real_attn_A.detach().cpu().squeeze(0)
                real_attn_B = real_attn_B.detach().cpu().squeeze(0)
                fake_attn_A = fake_attn_A.detach().cpu().squeeze(0)
                fake_attn_B = fake_attn_B.detach().cpu().squeeze(0)

                real_attn_A = heatmap_colorization(real_attn_A)
                real_attn_B = heatmap_colorization(real_attn_B)
                fake_attn_A = heatmap_colorization(fake_attn_A)
                fake_attn_B = heatmap_colorization(fake_attn_B)
                real_attn_A_image = 0.5*real_A_image+0.5*real_attn_A
                real_attn_B_image = 0.5*real_B_image+0.5*real_attn_B
                fake_attn_A_image = 0.5*fake_A_image+0.5*fake_attn_A
                fake_attn_B_image = 0.5*fake_B_image+0.5*fake_attn_B


            fig = plt.figure(); plt.axis("off");
            plt.title("Top: A->B / Bottom: B->A (epoch: {0:d})".format(epoch))
            if attention:
                plt.imshow(np.transpose(utils.make_grid([real_A_image,
                                                        real_attn_A_image,
                                                        fake_B_image,
                                                        fake_attn_B_image,
                                                        rect_A_image,
                                                        real_B_image,
                                                        real_attn_B_image,
                                                        fake_A_image,
                                                        fake_attn_A_image,
                                                        rect_B_image],
                                                        nrow=5,
                                                        padding=0,
                                                        normalize=True),
                                                        (1,2,0)))
            else:
                plt.imshow(np.transpose(utils.make_grid([real_A_image,
                                                        fake_B_image,
                                                        rect_A_image,
                                                        real_B_image,
                                                        fake_A_image,
                                                        rect_B_image],
                                                        nrow=3,
                                                        padding=0,
                                                        normalize=True),
                                                        (1,2,0)))
            IMAGE_NAME = 'G_AB_BA_{0}.png'.format(i)
            fig.savefig(os.path.join(SAVE_IMG_PATH, IMAGE_NAME),
                    bbox_inces='tight', pad_inches=0, dpi=150)
            plt.close()
            #===================================================================
            # Display current status
            print("[{:5d}/{:5d}]\t\t\r".format(i+1, total_iter), end='')
        print("Validation result images have been saved.")
