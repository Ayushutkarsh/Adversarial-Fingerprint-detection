import numpy as np
import cv2 as cv
import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torchvision.utils import save_image
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
import csv
import cv2
import random
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from PIL import Image
from scipy import ndimage

class Roi(object):
    def normalise(self,img):
        return (img - np.mean(img))/(np.std(img))


    def get_roi(self, im, w=8, threshold=.5):
        """
        Returns mask identifying the ROI. Calculates the standard deviation in each image block and threshold the ROI
        It also normalises the intesity values of
        the image so that the ridge regions have zero mean, unit standard
        deviation.
        :param im: Image
        :param w: size of the block
        :param threshold: std threshold
        :return: segmented_image
        """
        if len(im.shape)==2:
            (y, x) = im.shape
        elif len(im.shape)==3:
            (y, x, z) = im.shape
        else:
            print(" Image shape non matchable ---> ", im.shape)

        threshold = np.std(im)*threshold

        image_variance = np.zeros(im.shape)
        segmented_image = im.copy()
        mask = np.ones_like(im)

        for i in range(0, x, w):
            for j in range(0, y, w):
                box = [i, j, min(i + w, x), min(j + w, y)]
                block_stddev = np.std(im[box[1]:box[3], box[0]:box[2]])
                image_variance[box[1]:box[3], box[0]:box[2]] = block_stddev

        # apply threshold
        mask[image_variance < threshold] = 0

        # smooth mask with a open/close morphological filter
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(w*2, w*2))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        mask_seg = Image.fromarray(np.uint8(mask))
        mask_seg = mask_seg.convert('L')
        objs = ndimage.find_objects(mask_seg)

        return objs