#-*- coding:utf-8 -*-
'''
[AI502] Deep Learning Assignment
"Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" Implementation
20193640 Jungwon Choi
'''
import random

# Set the directory paths
FIGURE_PATH = './figures'
RESULT_PATH = './results/'
CHECKPOINT_PATH ='./checkpoints/'

#===============================================================================
class ImageBuffer():
    #===========================================================================
    ''' Initialization '''
    def __init__(self, buffer_size=50):
        self.buffer_size = buffer_size
        if self.buffer_size <= 0:
            raise Exception("buffer_size must be greater than zero.")
        self.num_images = 0
        self.image_buffer = []

    #===========================================================================
    ''' Return an image from the buffer. '''
    def query(self, image): # (batch size: 1)
        # buffer empty: just return input images
        if self.buffer_size == 0:
            return image

        # buffer not full: continue inserting image
        if self.num_images < self.buffer_size:
                self.num_images += 1
                self.image_buffer.append(image)
                return image

        # buffer full: by 50% chance, draw alot for image from buffer
        else:
            prob = random.uniform(0, 1)
            if prob > 0.5:
                random_idx = random.randint(0, self.buffer_size - 1)
                return_image = self.image_buffer[random_idx].clone()
                self.image_buffer[random_idx] = image
                return return_image
            else:
                return image
