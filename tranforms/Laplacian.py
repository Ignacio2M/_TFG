import random
import cv2 as cv
from tools.file_tools import ImageTools
from tranforms.ITransform import ITransform


class Laplacian(ITransform):
    file_tool: ImageTools

    def __init__(self, file_tool, random_apply):
        super(Laplacian, self).__init__()
        self.file_tool = file_tool
        self.random_apply = random_apply
        self.image_apply = []

    def apply(self):
        for index_sample, sample_dir in enumerate(self.file_tool.list_samples_dir):
            apply = []
            for index_image, name_image in enumerate(self.file_tool.list_items_from(sample_dir)):
                image = self.file_tool.load(name_image)
                if not self.random_apply is None:
                    if round(random.random(), 2) <= self.random_apply:
                        final_image = cv.Laplacian(image, cv.CV_64F)
                        apply.append(1)
                    else:
                        final_image = image
                        apply.append(0)
                else:
                    final_image = cv.Laplacian(image, cv.CV_64F)
                    apply.append(1)
                self.file_tool.save(name_image, final_image)
            self.image_apply.append(apply)

    def un_apply(self):
        pass

    def info(self):
        return {'type': "Laplacian",
                'apply_in': self.image_apply}
