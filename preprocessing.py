import cv2
import numpy as np
import math

class FastDenoiser: 
    """Denoises image by using CV's fastNlMeansDenoising method
    Params
    ------
    image       is the image to be Thresholded
    strength    the amount of denoising to apply
    Returns
    -------
    Denoised image
    """
    def __init__(self, strength = 7, output_process = False):
        self._strength = strength
        self.output_process = output_process


    def __call__(self, image):
        temp = cv2.fastNlMeansDenoising(image, h = self._strength)
        if self.output_process: cv2.imwrite('output/denoised.jpg', temp)
        return temp

class OtsuThresholder:
    def __init__(self, thresh1 = 0, thresh2 = 255, output_process = False):
        self.output_process = output_process
        self.thresh1 = thresh1
        self.thresh2 = thresh2


    def __call__(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        T_, thresholded = cv2.threshold(image, self.thresh1, self.thresh2, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if self.output_process: cv2.imwrite('output/thresholded.jpg', thresholded)
        return thresholded