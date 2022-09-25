import os
import re

import cv2 as cv


class ImageTools:
    original_dir: str
    original_dir_hr: str
    # preprocess_dir: str
    # measure_dir: str
    # net_dir: str
    # logs_dir: str
    # final_dir: str

    root_dir = "Data/Samples"
    measure_dir = "Measures"
    net_dir = 'Net'
    logs_dir = "Logs"
    final_dir = "Result"
    preprocess_dir = "Preprocess"
    preprocess_net_dir = "Preprocess_net"
    descript_dit = "Descript"

    CONTROL_NET_DIR = "Data/Final/Control/Net"
    CONTROL_MEASURE_DIR = "Data/Final/Control/Metrics"

    IMAGES_FILE_TYPE = ['.png', '.jpg']

    initial_shape: list or None

    def __init__(self, original_dir, original_dir_hr):
        self.control_metric_dir = None
        self.control_net_dir = None
        self.uuid = None
        self.initial_shape= None
        self.original_dir_hr = original_dir_hr
        self.original_dir = original_dir


        self.num_of_images = 0

    def built_structure(self, num_samples, uuid):
        self.uuid = uuid
        self.root_dir = self.create_dir(os.path.join(self.root_dir, uuid))
        self.measure_dir = self.create_dir(os.path.join(self.root_dir, self.measure_dir))
        self.logs_dir = self.create_dir(os.path.join(self.root_dir, self.logs_dir))
        self.preprocess_net_dir = self.create_dir(os.path.join(self.root_dir, self.preprocess_net_dir))
        self.net_dir = self.create_dir(os.path.join(self.root_dir, self.net_dir))
        self.preprocess_dir = self.create_dir(os.path.join(self.root_dir, self.preprocess_dir))
        self.descript_dit = self.create_dir(os.path.join(self.root_dir, self.descript_dit))
        self.final_dir = self.create_dir(os.path.join(self.root_dir, self.final_dir))

        self.control_net_dir = os.path.join(self.CONTROL_NET_DIR, os.path.split(self.original_dir)[-1])
        self.control_metric_dir = os.path.join(self.CONTROL_MEASURE_DIR, os.path.split(self.original_dir)[-1])

        for index_sample in range(num_samples):
            preprocess_dir_index_sample = os.path.join(self.preprocess_dir, str(index_sample))
            try:
                os.mkdir(preprocess_dir_index_sample)
            except FileExistsError:
                pass

        for image_name in os.listdir(self.original_dir):
            images_dir = os.path.join(self.original_dir, image_name)
            image = self.load(images_dir)
            if self.initial_shape is None:
                self.initial_shape = image.shape
            for index_sample in range(num_samples):
                sample_dir = os.path.join(self.preprocess_dir, str(index_sample))
                sample_dir = os.path.join(sample_dir, image_name)
                self.save(sample_dir, image)
        self.num_of_images = len(os.listdir(self.original_dir))

    def load(self, file_name):
        """
        raise TypeError if imagen tipy not define
        :param file_name:
        :return Image matrix:
        :raise TypeError:
        """
        file_type = os.path.splitext(file_name)[-1]
        if file_type in self.IMAGES_FILE_TYPE:
            return cv.imread(file_name)
        else:
            raise TypeError

    def save_preprocess_net_image(self, index, name, image):
        path = os.path.join(self.preprocess_net_dir, str(index))
        self.create_dir(path)
        self.save(os.path.join(path, name), image)

    def save_final(self, name, data, sub_dir=None):
        self.create_dir(self.final_dir)

        name = os.path.split(name)[-1]
        if sub_dir is None:
            final_dir = os.path.join(self.final_dir, name)
        else:
            aux_dir = os.path.join(self.final_dir, sub_dir)
            # compruebo si existe el directorio auxsiliar
            self.create_dir(aux_dir)
            final_dir = os.path.join(aux_dir, name)

        self.save(final_dir, data)

    def save(self, dir_name, data):
        file_type = os.path.splitext(dir_name)[-1]
        if file_type in self.IMAGES_FILE_TYPE:
            self._save_(dir_name, data)
        else:
            raise TypeError

    def _save_(self, path, data):
        cv.imwrite(path, data)


    def list_items_from(self, dir4list: str, regex=None) -> list:
        """
        Lista los elemntos del directorio **dir4list** filtrandola en caso de que se defina un patrn en **regex**

        :param dir4list: Directorio
        :param regex: Patron de regex
        :return: Lista finltrada.
        """
        list_items = list(map(lambda x: os.path.join(dir4list, x), os.listdir(dir4list)))
        if regex is not None:
            list_items = list(filter(lambda x: re.match(regex, x), list_items))
        return list_items

    @property
    def list_samples_dir(self) -> list:
        return self.list_items_from(self.preprocess_dir)

    @property
    def list_samples_net_dir(self) -> list:
        return self.list_items_from(self.net_dir)

    @property
    def list_samples_correct_net_dir(self) -> list:
        list_ = self.list_items_from(self.preprocess_net_dir)
        if len(list_) == 0:
            list_ = self.list_items_from(self.net_dir)
        return list_

    @property
    def exists_control_dir(self):
        return os.path.exists(self.CONTROL_NET_DIR) and len(os.listdir(self.CONTROL_NET_DIR)) > 0

    @property
    def exists_control_metrics(self):
        return os.path.exists(self.CONTROL_MEASURE_DIR) and len(os.listdir(self.CONTROL_MEASURE_DIR)) > 0

    def info(self):
        return {
            'original_dir' : self.original_dir,
            'original_dir_hr' : self.original_dir_hr,
            'preprocess_dir' : self.preprocess_dir,
            'measure_dir' : self.measure_dir,
            'net_dir' : self.net_dir,
            'logs_dir' : self.logs_dir,
            'final_dir' : self.final_dir,
            'control_net_dir' : self.CONTROL_NET_DIR,
            'control_metric_dir' : self.CONTROL_MEASURE_DIR
        }

    def create_dir(self, dir_):
        if not os.path.exists(dir_):
            os.mkdir(dir_)
        return dir_
