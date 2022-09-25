import random
import sys

import numpy as np
import cv2 as cv
from tools.file_tools import ImageTools
from tranforms.ITransform import ITransform
from skimage.util import random_noise


def gausian_noise(image, seed):
    return np.array(255 * random_noise(image, 'gaussian', var=0.001, seed=seed), dtype='uint8')

def salt_noise(image, seed):
    return np.array(255 * random_noise(image, 'salt', amount=0.001, seed=seed), dtype='uint8')

def sp_noise(image, seed):
    return np.array(255 * random_noise(image, 's&p', amount=0.001, seed=seed), dtype='uint8')

class Noise(ITransform):

    file_tool: ImageTools

    def __init__(self, file_tool, noise_type, random_apply):
        super(Noise, self).__init__()
        self.file_tool = file_tool
        self.noise_name = noise_type
        self.random_apply = random_apply
        self.image_apply = []
        self.seed_noise = []

    # ============ Noise =============
        if 'gaussian' == noise_type:
            self.noise = gausian_noise
        elif 'salt' == noise_type:
            self.noise = salt_noise
        elif 's&p' == noise_type:
            self.noise = sp_noise
        else:
            raise NotImplemented

    def apply(self):
        for index_sample, sample_dir in enumerate(self.file_tool.list_samples_dir):
            apply = []
            for index_image, name_image in enumerate(self.file_tool.list_items_from(sample_dir)):
                image = self.file_tool.load(name_image)
                self.seed_noise.append(random.randint(1, sys.maxsize))
                if not self.random_apply is None:
                    if round(random.random(), 2) <= self.random_apply:
                        final_image = self.noise(image, self.seed_noise[-1])
                        apply.append(1)
                    else:
                        final_image = image
                        apply.append(0)
                else:
                    final_image = self.noise(image, self.seed_noise[-1])
                    apply.append(1)
                self.file_tool.save(name_image, final_image)
            self.image_apply.append(apply)
    def un_apply(self):
        pass


    def info(self):
        return {'type': "RandomNoise",
                 'random_type': self.noise_name,
                 'apply_in': self.image_apply,
                 'random_seed': self.seed_noise}
