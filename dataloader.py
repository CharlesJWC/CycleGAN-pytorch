#-*- coding:utf-8 -*-
'''
[AI502] Deep Learning Assignment
"Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" Implementation
20193640 Jungwon Choi
'''
import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import os

#===============================================================================
''' CycleGAN Dataloader '''
class CycleGAN_Dataloader():
    #===========================================================================
    ''' Initialization '''
    def __init__(self, name, root=None, train=True, transform=None,
                                        batch_size=1, num_workers=8):
        # Load dataset
        self.dataset = Unpaierd_Dataset(name, root, train, transform)

        # Make dataLoader
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                batch_size=batch_size,
                                                shuffle=train,
                                                num_workers=num_workers,
                                                pin_memory=True)
    #===========================================================================
    ''' Return the number of dataset '''
    def __len__(self):
        return len(self.dataset)
    #===========================================================================
    ''' Return the batch of dataset '''
    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data

#===============================================================================
''' CycleGAN Unpaierd Dataset '''
class Unpaierd_Dataset():
    #===========================================================================
    # cityscapes    : real(A)       label(B)
    # horse2zebra   : horse(A)      zebra(B)
    # summer2winter : summer(A)     winter(B)
    # vangogh2photo : vangogh(A)    photo(B)
    #===========================================================================
    predefined_path = {
                        'cityscapes': './datasets/cityscapes',
                        'horse2zebra': './datasets/horse2zebra',
                        'summer2winter': './datasets/summer2winter_yosemite',
                        'vangogh2photo': './datasets/vangogh2photo',
                        'young2old': './datasets/young2old',
                        'man2woman': './datasets/man2woman',
                        'person2car': './datasets/person2car',
                        'young2old_500': './datasets/young2old_500',
                        'man2woman_500': './datasets/man2woman_500',
                        'person2car_500': './datasets/person2car_500',
                        'apple2orange': './datasets/apple2orange'
                      }
    #===========================================================================
    ''' Initialization '''
    def __init__(self, name, root=None, train=True, transform=None):
        self.train = train

        # Get dataset root path
        if root is None:
            if name in self.predefined_path:
                self.root = self.predefined_path[name]
            else:
                raise Exception("The dataset is not implemented.")
        else:
            self.root = root

        # Apply dataset transform
        if transform is None:
            if name == 'cityscapes':
                sequence =  [transforms.Resize((128, 128))]
            else:
                sequence =  [transforms.Resize((256, 256))]
            sequence += [transforms.ToTensor()]
            sequence += [transforms.Normalize((.5, .5, .5), (.5, .5, .5))]
            self.A_transform = transforms.Compose(sequence)
            self.B_transform = transforms.Compose(sequence)
        else:
            self.A_transform = transform
            self.B_transform = transform

        # Get image path list
        self.A_images = list()
        self.B_images = list()
        if self.train:
            for file_name in os.listdir(os.path.join(self.root, 'trainA')):
                self.A_images.append(os.path.join(self.root, 'trainA', file_name))
            for file_name in os.listdir(os.path.join(self.root, 'trainB')):
                self.B_images.append(os.path.join(self.root, 'trainB', file_name))
        else:
            for file_name in os.listdir(os.path.join(self.root, 'testA')):
                self.A_images.append(os.path.join(self.root, 'testA', file_name))
            for file_name in os.listdir(os.path.join(self.root, 'testB')):
                self.B_images.append(os.path.join(self.root, 'testB', file_name))
        self.A_images.sort()
        self.B_images.sort()

        # Get each dataset size
        self.A_size = len(self.A_images)
        self.B_size = len(self.B_images)
    #===========================================================================
    ''' Return the data '''
    def __getitem__(self, index):
        # Load images from the paths
        A_image = Image.open(self.A_images[index % self.A_size]).convert('RGB')
        B_image = Image.open(self.B_images[index % self.B_size]).convert('RGB')
        # Transform the images
        A_image = self.A_transform(A_image)
        B_image = self.B_transform(B_image)
        return A_image, B_image
    #===========================================================================
    ''' Return the size of dataset '''
    def __len__(self):
        # The larger one of A and B
        return max(self.A_size, self.B_size)


#===============================================================================
''' Check the dataset '''
if __name__ == '__main__':
    #===========================================================================
    import torchvision.utils as utils
    #===========================================================================

    dataset_list = ['cityscapes',
                    'horse2zebra',
                    'summer2winter',
                    'vangogh2photo',]

    for name in dataset_list:
        # Get images from dataloader
        dataloader = CycleGAN_Dataloader(name)
        for i, (A_image, B_image) in enumerate(dataloader):
            # Show images
            fig = plt.figure(); #figsize=(8,8) # plt.axis("off");
            plt.imshow(np.transpose(utils.make_grid([A_image[0], B_image[0]],
                                                    padding=2,
                                                    normalize=True),
                                                    (1,2,0)))
            fig.canvas.manager.window.move(0, 0)
            plt.show(block=False); plt.pause(1); plt.close()
            if i >= 2: break
