# CS4243 Assignment 1
# ===================
# Name: Tay Yang Shun
# Matric: A0073063M

import sys
import cv2
import cv2.cv as cv
import numpy as np

# IMAGE_FILE_NAME = sys.argv[1]
IMAGE_FILE_NAME = 'airborne.jpg'

image = cv2.imread(IMAGE_FILE_NAME, cv2.CV_LOAD_IMAGE_GRAYSCALE)
height, width = image.shape


flattened_image = image.reshape(1, -1)[0] # Convert to 1-d array

freq = np.bincount(flattened_image) # Calculate frequency of each intensity
freq = np.lib.pad(freq, (0, 256 - len(freq)), 'constant', constant_values=(0,0)) # Pad with zeroes
# print freq
cum_freq = np.cumsum(freq)

bin_size = sum(freq) // 256

# print bin_size
