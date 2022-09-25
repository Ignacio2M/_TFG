from matplotlib import pyplot as plt

from tranforms.ITransform import ITransform
from tools.file_tools import ImageTools
import numpy as np
import cv2 as cv

class VIS(ITransform):
    file_tool: ImageTools

    split_apply: list

    def __init__(self, file_tool, split_num, increment_step):
        super().__init__()
        self.block_size_y = None
        self.shape_y = None
        self.increment_step = increment_step
        self.file_tool = file_tool
        self.split_num = split_num
        self.split_index_array = None
        self.split_apply = []

    def apply(self):
        step = 0
        for sample_dir in self.file_tool.list_samples_dir:
            aux_list_split_pos = []
            for image_dir in self.file_tool.list_items_from(sample_dir):
                image = self.file_tool.load(image_dir)
                # ++++++++++++++ Generate split ++++++++++++++++++
                self.shape_y = image.shape[1]
                if self.split_index_array is None:
                    self.block_size_y = self.shape_y // self.split_num
                    self.split_index_array = np.array(
                        [(self.block_size_y + (increment*self.block_size_y)) % self.shape_y for increment in range(self.split_num)])
                    self.split_index_array.sort()
                split_index_list = ((self.split_index_array + (step*self.increment_step)) % self.shape_y).astype(int)
                aux_list_split_pos.append(split_index_list)
                image_proces = image
                for index, poss in enumerate(split_index_list):
                    index += 1
                    if index < image.shape[1]:
                        remplace = image[:, poss:poss + 1, :].copy()
                        random = np.random.rand(1)
                        remplace = np.multiply(remplace, random)
                        print(random)
                        image_proces = np.insert(image_proces, (index + poss),
                                                 (np.random.rand(3, image.shape[0], 1) * 255).astype(int).T, axis=1)
                        image_proces[:, (index + poss):(index + poss) + 1, :] = remplace
                        # image_proces = np.insert(image_proces, index+poss, (np.random.rand(3,image.shape[0],1)*255).astype(int).T, axis=1)
                step += 1
                # plt.imshow(image_proces)
                # plt.show()
                self.file_tool.save(image_dir, image_proces)
            self.split_apply.append(aux_list_split_pos)
        print(self.split_apply)


    def un_apply(self):
        pass
        # step = 0
        # for index_sample, sample_dir in enumerate(self.file_tool.list_samples_net_dir):
        #     for index_image, image_dir in enumerate(self.file_tool.list_items_from(sample_dir)):
        #         image = self.file_tool.load(image_dir)
        #         list_remove = []
        #         image_marquet = image.copy()
        #         image_proces = image.copy()
        #         for index, point_x in enumerate(self.split_apply[index_sample][index_image]):
        #             for _ in range(-2,3):
        #                 point_x_aux = (point_x * 4 + (4 * index))+_
        #                 if point_x_aux >= 0:
        #                     # list_remove.append(point_x_aux)
        #                     image_proces = np.delete(image_proces, point_x_aux, 1)
        #                     image_marquet = cv.line(image_marquet, (point_x_aux, 0), (point_x_aux, image.shape[0]),
        #                                             (0, 255, 0), 1)
        #         cv.imwrite('./basura/line.png', image_marquet)
        #         # image_proces = np.delete(image,list_remove, 1)
        #         self.file_tool.save(image_dir, image_proces)


    def info(self):
        return {'block_size_y': self.block_size_y,
                 'shape_y': self.shape_y,
                 'increment_step': self.increment_step,
                 'split_num': self.split_num}