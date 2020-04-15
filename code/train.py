#-*- coding:utf-8 -*-
'''
[AI502] Deep Learning Assignment
"Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" Implementation
20193640 Jungwon Choi
'''
import torch
import torch.nn as nn
import numpy as np

#===============================================================================
''' Train sequence '''
def train(train_dataloader, A_generator, B_generator, A_discriminator,
            B_discriminator, criterion_GAN, criterion_cycle, criterion_identity,
            optimizer_G, optimizer_D, A_buffer, B_buffer, loss_type,
            lambda_cycle, lambda_identity, criterion_feature, lambda_feature,
            attention):

    A_generator.train()
    B_generator.train()
    A_discriminator.train()
    B_discriminator.train()
    device = next(A_generator.parameters()).device.index
    losses_G = []
    losses_D_A = []
    losses_D_B = []
    total_iter = len(train_dataloader)

    for i, (A_image, B_image) in enumerate(train_dataloader):
        real_A_image, real_B_image = A_image.cuda(device), B_image.cuda(device)

        #-----------------------------------------------------------------------
        ''' Train Generator Networks '''
        #-----------------------------------------------------------------------
        # Set no grad for Discriminators
        for param in A_discriminator.parameters():
            param.requires_grad = False
        for param in B_discriminator.parameters():
            param.requires_grad = False

        # Generate each image (Forward)

        if attention:
            real_attn_A = B_discriminator.forward_attantionmap(real_A_image)
            fake_B_image = A_generator(real_attn_A*real_A_image) # G_A(A*attnA)
            fake_attn_B = A_discriminator.forward_attantionmap(fake_B_image)
            rect_A_image = B_generator(fake_attn_B*fake_B_image) # G_B(G_A(A*attnA)*attnB)

            real_attn_B = A_discriminator.forward_attantionmap(real_B_image)
            fake_A_image = A_generator(real_attn_B*real_B_image) # G_B(B*attnB)
            fake_attn_A = B_discriminator.forward_attantionmap(fake_A_image)
            rect_B_image = B_generator(fake_attn_A*fake_A_image) # G_A(G_B(B*attnB)*attnA)
        else:
            fake_B_image = A_generator(real_A_image)    # G_A(A)
            rect_A_image = B_generator(fake_B_image)    # G_B(G_A(A))

            fake_A_image = B_generator(real_B_image)    # G_B(B)
            rect_B_image = A_generator(fake_A_image)    # G_A(G_B(B))

        # Empty generators' gradients
        optimizer_G.zero_grad()

        # Calculate losses
        if loss_type not in ['Cycle_alone']:
            # GAN loss D_A(G_A(A))
            A_pred = A_discriminator(fake_B_image)
            true_label = torch.tensor(1.0).expand_as(A_pred).cuda(device)
            loss_G_A = criterion_GAN(A_pred, true_label)
            # GAN loss D_B(G_B(B))
            B_pred = B_discriminator(fake_A_image)
            true_label = torch.tensor(1.0).expand_as(B_pred).cuda(device)
            loss_G_B = criterion_GAN(B_pred, true_label)
        else:
            loss_G_A = 0
            loss_G_B = 0

        if loss_type not in ['GAN_alone', 'GAN_bwcycle']:
            # Forward cycle loss || G_B(G_A(A)) - A||
            loss_cycle_A = criterion_cycle(rect_A_image, real_A_image)
        else:
            loss_cycle_A = 0

        if loss_type not in ['GAN_alone', 'GAN_fwcycle']:
            # Backward cycle loss || G_A(G_B(B)) - B||
            loss_cycle_B = criterion_cycle(rect_B_image, real_B_image)
        else:
            loss_cycle_B = 0

        if lambda_identity > 0:
            # Identity loss A ||G_A(B) - B||
            idt_A_image = A_generator(real_B_image)
            loss_idt_A = criterion_identity(idt_A_image, real_B_image)
            # Identity loss B ||G_B(A) - A||
            idt_B_image = B_generator(real_A_image)
            loss_idt_B = criterion_identity(idt_B_image, real_A_image)
        else:
            loss_idt_A = 0
            loss_idt_B = 0

        if lambda_feature > 0:
            # feature map loss A ||G_A(A) - G_A(G_B(G_A(A)))||
            A_real_feature = A_generator.forward_featuremap(real_A_image)
            A_fake_feature = A_generator.forward_featuremap(rect_A_image)
            loss_fm_A = criterion_feature(A_real_feature, A_fake_feature)
            # feature map los B ||G_B(B) - G_B(G_A(G_B(B)))||
            B_real_feature = B_generator.forward_featuremap(real_B_image)
            B_fake_feature = B_generator.forward_featuremap(rect_B_image)
            loss_fm_B = criterion_feature(B_real_feature, B_fake_feature)
        else:
            loss_fm_A = 0
            loss_fm_B = 0

        # Combine all losses
        loss_G = loss_G_A + loss_G_B \
                + lambda_cycle*(loss_cycle_A + loss_cycle_B) \
                + lambda_identity*lambda_cycle*(loss_idt_A + loss_idt_B) \
                + lambda_feature*(loss_fm_A + loss_fm_B)
        losses_G.append(loss_G.item())

        # Calculate gradients (Backpropagation)
        loss_G.backward()

        # Update generators' parameters
        optimizer_G.step()

        #-----------------------------------------------------------------------
        ''' Train Generator Network '''
        #-----------------------------------------------------------------------
        # Set auto grad for Discriminators
        for param in A_discriminator.parameters():
            param.requires_grad = True
        for param in B_discriminator.parameters():
            param.requires_grad = True

        # Empty discriminators' gradients
        optimizer_D.zero_grad()

        # Get fake image from image buffer
        fake_A_image = A_buffer.query(fake_A_image)
        fake_B_image = B_buffer.query(fake_B_image)

        # Calculate losses

        # for real image
        A_pred_real = A_discriminator(real_B_image)
        true_label = torch.tensor(1.0).expand_as(A_pred_real).cuda(device)
        loss_D_A_real = criterion_GAN(A_pred_real, true_label)

        B_pred_real = B_discriminator(real_A_image)
        true_label = torch.tensor(1.0).expand_as(B_pred_real).cuda(device)
        loss_D_B_real = criterion_GAN(B_pred_real, true_label)

        # for fake image
        A_pred_fake = A_discriminator(fake_B_image.detach())
        fake_label = torch.tensor(0.0).expand_as(A_pred_fake).cuda(device)
        loss_D_A_fake = criterion_GAN(A_pred_fake, fake_label)

        B_pred_fake = B_discriminator(fake_A_image.detach())
        fake_label = torch.tensor(0.0).expand_as(B_pred_fake).cuda(device)
        loss_D_B_fake = criterion_GAN(B_pred_fake, fake_label)

        # Combine losses
        loss_D_A = 0.5 * (loss_D_A_real + loss_D_A_fake)
        loss_D_B = 0.5 * (loss_D_B_real + loss_D_B_fake)
        losses_D_A.append(loss_D_A.item())
        losses_D_B.append(loss_D_B.item())

        # Calculate gradients (Backpropagation)
        loss_D_A.backward()
        loss_D_B.backward()

        # Update discriminators' parameters
        optimizer_D.step()

        #-----------------------------------------------------------------------
        # Display current status
        print("[{:5d}/{:5d}]\t\t\r".format(i+1, total_iter), end='')
    #===========================================================================
    avg_loss_G = sum(losses_G)/len(losses_G)
    avg_loss_D_A = sum(losses_D_A)/len(losses_D_A)
    avg_loss_D_B = sum(losses_D_B)/len(losses_D_B)
    avg_loss_D = {'A':avg_loss_D_A, 'B':avg_loss_D_B}
    return avg_loss_G, avg_loss_D
