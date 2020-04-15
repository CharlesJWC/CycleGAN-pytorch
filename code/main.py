    #-*- coding:utf-8 -*-
'''
[AI502] Deep Learning Assignment
"Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" Implementation
20193640 Jungwon Choi
'''
'''
# I referenced the authors code for Implementation details as below links,
# but I coded whole parts myself, not copy and paste.
# Reference : https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import itertools
import argparse
import random
import pickle
import time
import os
import numpy as np

# Implementation files
from dataloader import CycleGAN_Dataloader
from models.Generator import Generator
from models.Discriminator import Discriminator
from train import train
from val import val
from util import ImageBuffer
from util import RESULT_PATH, CHECKPOINT_PATH

VERSION_CHECK_MESSAGE = 'NOW 19-12-02 23:51'

#===============================================================================
''' Experiment1 : Ablation study of the CycleGAN Loss '''
    # Check the importance of both the GAN loss and the cycle consistency loss
''' Experiment2 : Train each CycleGAN on different unpaired datasets '''
    # Check the generality of CycleGAN on where paired data does not exist
def main(args):
    # Step0 ====================================================================
    # Set GPU ids
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_ids

    # Set the file name format
    FILE_NAME_FORMAT = "{0}_{1}_{2}_{3:d}{4}".format(
                    args.model, args.dataset, args.loss, args.epochs, args.flag)
    # Set the results file path
    RESULT_FILE_NAME = FILE_NAME_FORMAT+'_results.pkl'
    RESULT_FILE_PATH = os.path.join(RESULT_PATH, RESULT_FILE_NAME)
    # Set the checkpoint file path
    CHECKPOINT_FILE_NAME = FILE_NAME_FORMAT+'.ckpt'
    CHECKPOINT_FILE_PATH = os.path.join(CHECKPOINT_PATH, CHECKPOINT_FILE_NAME)
    BEST_CHECKPOINT_FILE_NAME = FILE_NAME_FORMAT+'_best.ckpt'
    BEST_CHECKPOINT_FILE_PATH = os.path.join(CHECKPOINT_PATH,
                                                    BEST_CHECKPOINT_FILE_NAME)
    # Set the random seed same for reproducibility
    random.seed(190811)
    torch.manual_seed(190811)
    torch.cuda.manual_seed_all(190811)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Step1 ====================================================================
    # Load dataset
    train_dataloader = CycleGAN_Dataloader(name=args.dataset,
                                     num_workers=args.num_workers)
    test_dataloader = CycleGAN_Dataloader(name=args.dataset, train=False,
                                     num_workers=args.num_workers)
    print('==> DataLoader ready.')

    # Step2 ====================================================================
    # Make the model
    if args.dataset == 'cityscapes':
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

    # Step3 ====================================================================
    # Set each loss function
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    criterion_feature = nn.L1Loss()

    # Set each optimizer
    optimizer_G = optim.Adam(itertools.chain(A_generator.parameters(),
                                    B_generator.parameters()),
                                    lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(itertools.chain(A_discriminator.parameters(),
                                    B_discriminator.parameters()),
                                    lr=args.lr, betas=(0.5, 0.999))
    # Set learning rate scheduler
    def lambda_rule(epoch):
        epoch_decay=args.epochs/2
        lr_linear_scale = 1.0 - max(0, epoch + 1 - epoch_decay) \
                                / float(epoch_decay+ 1)
        return lr_linear_scale
    scheduler_G = lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
    scheduler_D = lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_rule)
    print('==> Criterion and optimizer ready.')

    # Step4 ====================================================================
    # Train and validate the model
    start_epoch = 0
    best_metric = float("inf")

    # Initialize the result lists
    train_loss_G = []
    train_loss_D_A = []
    train_loss_D_B = []

    # Set image buffer
    A_buffer = ImageBuffer(args.buffer_size)
    B_buffer = ImageBuffer(args.buffer_size)

    if args.resume:
        assert os.path.exists(CHECKPOINT_FILE_PATH), 'No checkpoint file!'
        checkpoint = torch.load(CHECKPOINT_FILE_PATH)
        A_generator.load_state_dict(checkpoint['A_generator_state_dict'])
        B_generator.load_state_dict(checkpoint['B_generator_state_dict'])
        A_discriminator.load_state_dict(checkpoint['A_discriminator_state_dict'])
        B_discriminator.load_state_dict(checkpoint['B_discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
        scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
        start_epoch = checkpoint['epoch']
        train_loss_G = checkpoint['train_loss_G']
        train_loss_D_A = checkpoint['train_loss_D_A']
        train_loss_D_B = checkpoint['train_loss_D_B']
        best_metric = checkpoint['best_metric']

    # Save the training information
    result_data = {}
    result_data['model']            = args.model
    result_data['dataset']          = args.dataset
    result_data['loss']             = args.loss
    result_data['target_epoch']     = args.epochs
    result_data['batch_size']       = args.batch_size

    # Check the directory of the file path
    if not os.path.exists(os.path.dirname(RESULT_FILE_PATH)):
        os.makedirs(os.path.dirname(RESULT_FILE_PATH))
    if not os.path.exists(os.path.dirname(CHECKPOINT_FILE_PATH)):
        os.makedirs(os.path.dirname(CHECKPOINT_FILE_PATH))
    print('==> Train ready.')

    for epoch in range(args.epochs):
        # strat after the checkpoint epoch
        if epoch < start_epoch:
            continue

        print("\n[Epoch: {:3d}/{:3d}]".format(epoch+1, args.epochs))
        epoch_time = time.time()
        #=======================================================================
        # train and validate the model
        tloss_G, tloss_D = train(train_dataloader, A_generator, B_generator,
                            A_discriminator, B_discriminator,
                            criterion_GAN, criterion_cycle, criterion_identity,
                            optimizer_G, optimizer_D, A_buffer, B_buffer,
                            args.loss, args.lambda_cycle, args.lambda_identity,
                            criterion_feature, args.lambda_feature,
                            args.attention)
        train_loss_G.append(tloss_G)
        train_loss_D_A.append(tloss_D['A'])
        train_loss_D_B.append(tloss_D['B'])

        if (epoch+1) % 10 == 0:
            val(test_dataloader, A_generator, B_generator,
                A_discriminator, B_discriminator, epoch+1,
                FILE_NAME_FORMAT, args.attention)

        # Update the optimizer's learning rate
        current_lr = optimizer_G.param_groups[0]['lr']
        scheduler_G.step()
        scheduler_D.step()
        #=======================================================================
        current = time.time()

        # Save the current result
        result_data['current_epoch']    = epoch
        result_data['train_loss_G']     = train_loss_G
        result_data['train_loss_D_A']   = train_loss_D_A
        result_data['train_loss_D_B']   = train_loss_D_B

        # Save result_data as pkl file
        with open(RESULT_FILE_PATH, 'wb') as pkl_file:
            pickle.dump(result_data, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

        # Save the best checkpoint
        # if train_loss_G < best_metric:
        #     best_metric = train_loss_G
        #     torch.save({
        #         'epoch': epoch+1,
        #         'A_generator_state_dict': A_generator.state_dict(),
        #         'B_generator_state_dict': B_generator.state_dict(),
        #         'A_discriminator_state_dict': A_discriminator.state_dict(),
        #         'B_discriminator_state_dict': B_discriminator.state_dict(),
        #         'optimizer_G_state_dict': optimizer_G.state_dict(),
        #         'optimizer_D_state_dict': optimizer_D.state_dict(),
        #         'scheduler_G_state_dict': scheduler_G.state_dict(),
        #         'scheduler_D_state_dict': scheduler_D.state_dict(),
        #         'train_loss_G': train_loss_G,
        #         'train_loss_D_A': train_loss_D_A,
        #         'train_loss_D_B': train_loss_D_B,
        #         'best_metric': best_metric,
        #         }, BEST_CHECKPOINT_FILE_PATH)

        # Save the current checkpoint
        torch.save({
            'epoch': epoch+1,
            'A_generator_state_dict': A_generator.state_dict(),
            'B_generator_state_dict': B_generator.state_dict(),
            'A_discriminator_state_dict': A_discriminator.state_dict(),
            'B_discriminator_state_dict': B_discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'scheduler_G_state_dict': scheduler_G.state_dict(),
            'scheduler_D_state_dict': scheduler_D.state_dict(),
            'train_loss_G': train_loss_G,
            'train_loss_D_A': train_loss_D_A,
            'train_loss_D_B': train_loss_D_B,
            'best_metric': best_metric,
            }, CHECKPOINT_FILE_PATH)

        if (epoch+1) % 10 == 0:
            CHECKPOINT_FILE_NAME_epoch = FILE_NAME_FORMAT+'_{0}.ckpt'
            CHECKPOINT_FILE_PATH_epoch = os.path.join(CHECKPOINT_PATH,
                                                    FILE_NAME_FORMAT,
                                                    CHECKPOINT_FILE_NAME_epoch)
            if not os.path.exists(os.path.dirname(CHECKPOINT_FILE_PATH_epoch)):
                os.makedirs(os.path.dirname(CHECKPOINT_FILE_PATH_epoch))
            torch.save({
                'epoch': epoch+1,
                'A_generator_state_dict': A_generator.state_dict(),
                'B_generator_state_dict': B_generator.state_dict(),
                'A_discriminator_state_dict': A_discriminator.state_dict(),
                'B_discriminator_state_dict': B_discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'scheduler_G_state_dict': scheduler_G.state_dict(),
                'scheduler_D_state_dict': scheduler_D.state_dict(),
                'train_loss_G': train_loss_G,
                'train_loss_D_A': train_loss_D_A,
                'train_loss_D_B': train_loss_D_B,
                'best_metric': best_metric,
                }, CHECKPOINT_FILE_PATH_epoch)

        # Print the information on the console
        print("model                : {}".format(args.model))
        print("dataset              : {}".format(args.dataset))
        print("loss                 : {}".format(args.loss))
        print("batch_size           : {}".format(args.batch_size))
        print("current lrate        : {:f}".format(current_lr))
        print("G loss               : {:f}".format(tloss_G))
        print("D A/B loss           : {:f}/{:f}".format(tloss_D['A'], tloss_D['B']))
        print("epoch time           : {0:.3f} sec".format(current - epoch_time))
        print("Current elapsed time : {0:.3f} sec".format(current - start))
    print('==> Train done.')

    print(' '.join(['Results have been saved at', RESULT_FILE_PATH]))
    print(' '.join(['Checkpoints have been saved at', CHECKPOINT_FILE_PATH]))

#===============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CycleGAN Implementation')
    parser.add_argument('--model', default='CycleGAN', type=str)
    parser.add_argument('--dataset', default='horse2zebra', type=str,
            help='cityscapes, horse2zebra, summer2winter, vangogh2photo')
    parser.add_argument('--loss', default='CycleGAN', type=str,
            help='CycleGAN, Cycle_alone, GAN_alone, GAN_fwcycle, GAN_bwcycle')
    parser.add_argument('--lambda_cycle', default=10.0, type=float)
    parser.add_argument('--lambda_identity', default=0.0, type=float)
    parser.add_argument('--lambda_feature', default=0.0, type=float)
    parser.add_argument('--attention', action='store_true')
    parser.add_argument('--buffer_size', default=50, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--lr', default=0.0002, type=float)
    parser.add_argument('--gpu_ids', default='0', type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--flag', default='', type=str)
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    # Code version check message
    print(VERSION_CHECK_MESSAGE)

    start = time.time()
    #===========================================================================
    main(args)
    #===========================================================================
    end = time.time()
    print("Total elapsed time: {0:.3f} sec\n".format(end - start))
    print("[Finih time]",time.strftime('%c', time.localtime(time.time())))
