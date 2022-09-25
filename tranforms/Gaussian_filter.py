import random

from tranforms.ITransform import ITransform
import cv2 as cv


class Gaussian(ITransform):
    def __init__(self, file_tool, kernel, sigma_x, random_apply):
        super().__init__()
        self.sigma_x = sigma_x
        self.kernel = kernel
        self.file_tool = file_tool
        self.random_apply = random_apply

    def apply(self):
        for index_sample, sample_dir in enumerate(self.file_tool.list_samples_dir):
            for index_image, name_image in enumerate(self.file_tool.list_items_from(sample_dir)):
                print('Sample{}, Image: {}'.format(index_sample, index_image))
                # _____________ Transform ________________
                image = self.file_tool.load(name_image)
                if self.random_apply is not None:
                    if round(random.random(), 2) <= self.random_apply:
                        final_image = cv.GaussianBlur(image, self.kernel, self.sigma_x)
                        print('Gaussian::Sample{}, Image: {}'.format(index_sample, index_image))
                    else:
                        final_image = image
                else:
                    final_image = cv.GaussianBlur(image, self.kernel, self.sigma_x)

                self.file_tool.save(name_image, final_image)
        print("")

    def un_apply(self):
        pass
    
    def info(self):
        return { 'type': "Filtros_Gausiano",
                'sigma_x': str(self.sigma_x),
                 'kernel': str(self.kernel),
                 'random_apply': str(self.random_apply)}
