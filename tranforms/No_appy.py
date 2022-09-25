from matplotlib import pyplot as plt

from tranforms.ITransform import ITransform
from tools.file_tools import ImageTools
import numpy as np
import cv2 as cv

class No_apply(ITransform):
    file_tool: ImageTools

    split_apply: list

    def __init__(self):
        super().__init__()

    def apply(self):
        pass


    def un_apply(self):
        pass



    def info(self):
        return {}