import logging

from tools.file_tools import ImageTools


class UnApplyBlock:
    ID = 0
    id: int
    count = 0
    max_samples: int

    def __init__(self, file_tool: ImageTools, num_samples, id = None):
        UnApplyBlock.ID += 1
        self.id = id if not id is None else UnApplyBlock.ID
        self.max_samples = num_samples
        self.file_tool = file_tool
        self.final_images = [None] * self.file_tool.num_of_images

    def process(self,num_sample, index_image, image):
        if self.max_samples > num_sample:
            if self.final_images[index_image] is None:
                self.final_images[index_image] = image / self.max_samples
            else:
                self.final_images[index_image] = self.final_images[index_image] + (
                        image / self.max_samples)
            logging.info(
                'BLOCK{NAME} add image:{IMAGE}{MAS_SAMPLES}'.format(NAME=self.id, IMAGE=num_sample, MAS_SAMPLES=self.max_samples))

    def save_un_apply(self):
        for index, image in enumerate(self.final_images):
            self.file_tool.save_final(str(index) + '.png', image, sub_dir='Sample_{}({})'.format(self.id, self.max_samples))

