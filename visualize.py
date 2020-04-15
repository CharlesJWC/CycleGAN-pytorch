#-*- coding:utf-8 -*-
'''
[AI502] Deep Learning Assignment
"Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" Implementation
20193640 Jungwon Choi
'''
import os
import sys
import time
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
# import seaborn as sns

# import PIL
# from PIL import Image
# from scipy.signal import savgol_filter

import torch
import torch.nn as nn
import torchvision.utils as utils

from dataloader import CycleGAN_Dataloader
from models.Generator import Generator
from models.Discriminator import Discriminator
from val import val
from util import FIGURE_PATH, RESULT_PATH, CHECKPOINT_PATH


os.environ["CUDA_VISIBLE_DEVICES"]="3"

#===============================================================================
TEST_CHECKPOINT_LIST = [
    #---------------------------------------------------------------------------
    'CycleGAN_cityscapes_CycleGAN_200.ckpt',
    'CycleGAN_cityscapes_Cycle_alone_200.ckpt',
    'CycleGAN_cityscapes_GAN_alone_200.ckpt',
    'CycleGAN_cityscapes_GAN_fwcycle_200.ckpt',
    'CycleGAN_cityscapes_GAN_bwcycle_200.ckpt',
    #---------------------------------------------------------------------------
    'CycleGAN_horse2zebra_CycleGAN_200.ckpt',
    'CycleGAN_apple2orange_CycleGAN_200.ckpt',
    'CycleGAN_summer2winter_CycleGAN_200.ckpt',
    'CycleGAN_vangogh2photo_CycleGAN_200.ckpt',
    #---------------------------------------------------------------------------
    'CycleGAN_man2woman_CycleGAN_200.ckpt',
    'CycleGAN_person2car_CycleGAN_200.ckpt',
    #---------------------------------------------------------------------------
    'CycleGAN_apple2orange_CycleGAN_200_both.ckpt',
    'CycleGAN_apple2orange_CycleGAN_200_attention.ckpt',
    'CycleGAN_apple2orange_CycleGAN_200_feature.ckpt',
    'CycleGAN_person2car_500_CycleGAN_200_dt500_both.ckpt',
    'CycleGAN_person2car_500_CycleGAN_200_dt500_attention.ckpt',
    'CycleGAN_person2car_500_CycleGAN_200_dt500_feature.ckpt',
    #---------------------------------------------------------------------------
]

PKL_FILE_LIST = [
    #---------------------------------------------------------------------------
    'CycleGAN_cityscapes_CycleGAN_200_results.pkl',
    'CycleGAN_cityscapes_Cycle_alone_200_results.pkl',
    'CycleGAN_cityscapes_GAN_alone_200_results.pkl',
    'CycleGAN_cityscapes_GAN_fwcycle_200_results.pkl',
    'CycleGAN_cityscapes_GAN_bwcycle_200_results.pkl',
    #---------------------------------------------------------------------------
    'CycleGAN_horse2zebra_CycleGAN_200_results.pkl',
    'CycleGAN_apple2orange_CycleGAN_200_results.pkl',
    'CycleGAN_summer2winter_CycleGAN_200_results.pkl',
    'CycleGAN_vangogh2photo_CycleGAN_200_results.pkl',
    #---------------------------------------------------------------------------
    'CycleGAN_man2woman_CycleGAN_200_results.pkl',
    'CycleGAN_person2car_CycleGAN_200_results.pkl',
    #---------------------------------------------------------------------------
    'CycleGAN_apple2orange_CycleGAN_200_both_results.pkl',
    'CycleGAN_apple2orange_CycleGAN_200_attention_results.pkl',
    'CycleGAN_apple2orange_CycleGAN_200_feature_results.pkl',
    'CycleGAN_person2car_500_CycleGAN_200_dt500_both_results.pkl',
    'CycleGAN_person2car_500_CycleGAN_200_dt500_attention_results.pkl',
    'CycleGAN_person2car_500_CycleGAN_200_dt500_feature_results.pkl',
    #---------------------------------------------------------------------------
]

#===============================================================================
''' Visualize test images with the trained model '''
def visualize_test_images(ckpt_list):
    #===========================================================================
    for ckpt_name in ckpt_list:
        try:
            # Step0 ============================================================
            # Parsing the hyper-parameters
            FILE_NAME_FORMAT = ckpt_name.split('.')[0]
            parsing_list = ckpt_name.split('.')[0].split('_')

            # Setting constants
            model_name            = parsing_list[0]
            dataset_name          = parsing_list[1]
            loss_type             = parsing_list[2]
            flag                  = parsing_list[-1]

            if 'attention' in flag:
                attention = True
            else:
                attention = False
            # Step1 ============================================================
            # Load dataset
            test_dataloader = CycleGAN_Dataloader(name=dataset_name,
                                                  train=False,
                                                  num_workers=8)
            print('==> DataLoader ready.')

            # Step2 ============================================================
            # Make the model
            if dataset_name == 'cityscapes':
                A_generator       = Generator(num_resblock=6)
                B_generator       = Generator(num_resblock=6)
                A_discriminator   = Discriminator()
                B_discriminator   = Discriminator()
            else:
                A_generator       = Generator(num_resblock=9)
                B_generator       = Generator(num_resblock=9)
                A_discriminator   = Discriminator()
                B_discriminator   = Discriminator()

            # Check DataParallel available
            if torch.cuda.device_count() > 1:
                A_generator = nn.DataParallel(A_generator)
                B_generator = nn.DataParallel(B_generator)
                A_discriminator = nn.DataParallel(A_discriminator)
                B_discriminator = nn.DataParallel(B_discriminator)

            # Check CUDA available
            if torch.cuda.is_available():
                A_generator.cuda()
                B_generator.cuda()
                A_discriminator.cuda()
                B_discriminator.cuda()
            print('==> Model ready.')

            # Step3 ============================================================
            # Test the model
            checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, ckpt_name))
            A_generator.load_state_dict(checkpoint['A_generator_state_dict'])
            B_generator.load_state_dict(checkpoint['B_generator_state_dict'])
            A_discriminator.load_state_dict(checkpoint['A_discriminator_state_dict'])
            B_discriminator.load_state_dict(checkpoint['B_discriminator_state_dict'])
            train_epoch = checkpoint['epoch']

            val(test_dataloader, A_generator, B_generator,
                A_discriminator, B_discriminator, train_epoch,
                FILE_NAME_FORMAT, attention)

            #-------------------------------------------------------------------
            # Print the result on the console
            print("model   : {}".format(model_name))
            print("dataset : {}".format(dataset_name))
            print("loss    : {}".format(loss_type))
            print('-'*50)
        except Exception as e:
            print(e)
    print('==> Visualize test images done.')

#===============================================================================
def visualize_loss_graph(plk_file_list):
    for plk_file_name in plk_file_list:
        try:
            #===================================================================
            # Load results data
            plk_file_path = os.path.join(RESULT_PATH, plk_file_name)
            with open(plk_file_path, 'rb') as pkl_file:
                result_dict = pickle.load(pkl_file)

            train_loss_G = result_dict['train_loss_G']
            train_loss_D_A = result_dict['train_loss_D_A']
            train_loss_D_B = result_dict['train_loss_D_B']
            #===================================================================
            # Save figure
            FILE_NAME_FORMAT = '_'.join(os.path.splitext(plk_file_name)[0].split('_')[:-1])
            model_type = FILE_NAME_FORMAT.split('_')[0]
            SAVE_IMG_PATH = os.path.join(FIGURE_PATH, FILE_NAME_FORMAT)

            # Check the directory of the file path
            if not os.path.exists(SAVE_IMG_PATH):
                os.makedirs(SAVE_IMG_PATH)

            num_epoch = len(train_loss_G)
            epochs = np.arange(1, num_epoch+1)

            #-------------------------------------------------------------------
            # Generator Loss Graph
            #-------------------------------------------------------------------
            fig = plt.figure(dpi=150)
            plt.title('Generator Loss'), plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.plot(epochs, train_loss_G,'g-', markersize=1, alpha=0.8, label='G')
            plt.xlim([1, num_epoch])
            plt.legend()
            file_name = "Loss_G_graph.png"
            fig.savefig(os.path.join(SAVE_IMG_PATH, file_name), format='png')
            plt.close()

            #-------------------------------------------------------------------
            # Discriminator Loss Graph
            #-------------------------------------------------------------------
            fig = plt.figure(dpi=150)
            plt.title('Discriminator Loss'), plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.plot(epochs, train_loss_D_A,'-', markersize=1, alpha=0.8, label='D_A')
            plt.plot(epochs, train_loss_D_B,'-', markersize=1, alpha=0.8, label='D_B')
            plt.xlim([1, num_epoch])
            plt.legend()
            file_name = "Loss_D_graph.png"
            fig.savefig(os.path.join(SAVE_IMG_PATH, file_name), format='png')
            plt.close()
        except Exception as e:
            print(e)
    print('==> Loss graph visualization done.')

#===============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--loss', action='store_true')
    args = parser.parse_args()
    if args.test:
        visualize_test_images(TEST_CHECKPOINT_LIST)
    if args.loss:
        visualize_loss_graph(PKL_FILE_LIST)
    pass
