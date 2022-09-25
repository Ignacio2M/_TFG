import os

import numpy as np
from matplotlib import pyplot as plt

from tools.file_tools import ImageTools
import cv2 as cv

from tranforms.ITransform import ITransform


def final_corners(image, matrix):
    image_shape = image.shape
    init_point = np.array([[0, 0, 1],
                           [image_shape[1], 0, 1],
                           [image_shape[1], image_shape[0], 1],
                           [0, image_shape[0], 1],
                           ]).T

    final_point = (matrix @ init_point).astype(int)
    return final_point


def apply_transform(image, final_shape: np.array, angle, traslation, correct_matrix, borderMode):
    img_height, img_width = image.shape[0], image.shape[1]

    centre_y, centre_x = img_height // 2, img_width // 2

    matrix = cv.getRotationMatrix2D((centre_x, centre_y), angle, 1.0)

    matrix[0][2] += traslation[0]
    matrix[1][2] += traslation[1]

    matrix = matrix + correct_matrix
    image = cv.warpAffine(image, matrix, (final_shape[0], final_shape[1]), flags=cv.WARP_FILL_OUTLIERS,
                          borderMode= None if borderMode else cv.BORDER_REPLICATE)
    return matrix, image


def correct_image(image, max_angle, max_shift):
    img_height, img_width = image.shape[0], image.shape[1]

    centre_y, centre_x = img_height // 2, img_width // 2

    matrix = cv.getRotationMatrix2D((centre_x, centre_y), max_angle, 1.0)

    final_point = final_corners(image, matrix)
    shift = np.array((abs(np.min(final_point[0]) if np.min(final_point[0]) < 0 else 0),
                      abs(np.min(final_point[1]) if np.min(final_point[1]) < 0 else 0)))

    matrix[0][2] -= shift[0]
    matrix[1][2] -= shift[1]

    correct_matrix = np.array(
        [[0, 0, 0],
         [0, 0, centre_x - centre_y]]
    )

    final_shape = np.array([max(img_height, img_width), max(img_height, img_width)]) + max_shift

    return correct_matrix, final_shape


class SR(ITransform):
    shift_vector: np.ndarray
    angle_time_const: bool
    shift_time_const: bool
    angle_space_const: bool
    shift_space_const: bool
    matrix_transform: list
    file_tool: ImageTools

    def __init__(self, file_tool,
                 shift_vector: np.array,
                 angle_time_const=False,
                 shift_time_const=True,
                 angle_space_const=True,
                 shift_space_const=False,
                 black_backraun = False):
        super(SR, self).__init__()
        self.file_tool = file_tool
        if not isinstance(shift_vector, np.ndarray):
            if isinstance(shift_vector, list):
                shift_vector = np.array(shift_vector)
            else:
                raise TypeError("shift_vector deve de ser de tipo np.ndarray")
        self.shift_vector = shift_vector
        self.matrix_transform = []
        self.black_backraun = black_backraun

        # 0101 -> No
        self.shift_time_const = shift_time_const
        self.angle_time_const = angle_time_const
        self.angle_space_const = angle_space_const
        self.shift_space_const = shift_space_const

    def apply(self):
        # conatodr = 1
        # conatodr_c = 1

        sample_rotation = 0
        sample_shift = np.array([0, 0])

        last_init = np.array([0, 0])
        max_angle = 0

        if not self.angle_time_const or not self.angle_space_const:
            max_angle = 90

        if not self.shift_time_const:
            max_iteration = self.file_tool.num_of_images
        else:
            max_iteration = 0
        if not self.shift_space_const:
            max_iteration += len(self.file_tool.list_samples_dir) - 1

        max_shift = self.shift_vector * max_iteration

        for index_sample, sample_dir in enumerate(self.file_tool.list_samples_dir):
            aux_matrix_list = []
            for index_image, name_image in enumerate(self.file_tool.list_items_from(sample_dir)):
                print('Sample{}, Image: {}'.format(index_sample, index_image))
                # _____________ Transform ________________
                image = self.file_tool.load(name_image)
                correct_matrix, final_shape = correct_image(image, max_angle, max_shift)

                matrix, final_image = apply_transform(image, final_shape, sample_rotation, sample_shift,
                                                      correct_matrix, self.black_backraun)

                self.file_tool.save(name_image, final_image)
                aux_matrix_list.append(matrix)

                # # ------------ Constant_Time ------------
                if not self.angle_time_const:
                    sample_rotation += 90

                if not self.shift_time_const:
                    sample_shift += self.shift_vector

            self.matrix_transform.append(aux_matrix_list)

            # ------------ Constant_space ------------
            if not self.angle_space_const:
                sample_rotation += 90
            else:
                sample_rotation = 0

            if not self.shift_space_const:
                sample_shift = last_init + self.shift_vector
                last_init = sample_shift.copy()
            else:
                sample_shift = np.array([0, 0])

    def un_apply(self):

        for index_dir, dir_name in enumerate(self.file_tool.list_samples_net_dir):

            dir_name: str

            transform = self.matrix_transform

            final_image_list = []
            for index_image, images_path in enumerate(self.file_tool.list_items_from(dir_name)):
                image = self.file_tool.load(images_path)
                matrix = np.array(transform[index_dir][index_image])
                initial_shape = self.file_tool.initial_shape
                initial_shape = (initial_shape[1], initial_shape[0])

                init_point = np.array([
                    [0, 0, 1],
                    [0, initial_shape[1], 1],
                    [initial_shape[0], 0, 1]
                ]).astype(np.float32)

                # print("================================================================")

                # print(matrix)

                final_point = np.array(list(map(lambda x: (matrix @ x), init_point))).astype(np.float32)
                matrix = cv.getAffineTransform(final_point * 4, init_point[:, :2].astype(np.float32) * 4)
                # plt.imshow(image)
                # plt.show()
                image = cv.warpAffine(image, matrix, (
                initial_shape[0] * 4, initial_shape[1] * 4))  # (initial_shape[0] * 4, initial_shape[1] * 4)
                # plt.imshow(image)
                # plt.show()
                name = os.path.split(images_path)[-1]
                self.file_tool.save_preprocess_net_image(index_dir, name, image)

    def info(self):
        return {
             'type': "SR",
             'shift_vector': str(self.shift_vector),
             'num_of_matrix_transform': str(len(self.matrix_transform)),
             'shift_time_const': self.shift_time_const,
             'angle_time_const': self.angle_time_const,
             'angle_space_const': self.angle_space_const,
             'shift_space_const': self.shift_space_const}

