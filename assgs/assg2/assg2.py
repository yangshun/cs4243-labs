# CS4243 Assignment 2
# ===================
# Name: Tay Yang Shun
# Matric: A0073063M

import sys
import cv2
import cv2.cv as cv
import numpy as np

# Example usage: 
# python assg2.py labPhoto.JPG

IMAGE_FILE_NAME = sys.argv[1]
image = cv2.imread(IMAGE_FILE_NAME, cv2.CV_LOAD_IMAGE_GRAYSCALE)
