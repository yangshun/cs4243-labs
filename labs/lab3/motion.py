# CS4243 Lab 3
# ===================
# Name: Tay Yang Shun
# Matric: A0073063M

import cv2
import cv2.cv as cv

import numpy as numpy

im = cv2.imread('LabPhoto1.jpg', cv2.CV_LOAD_IMAGE_COLOR)
gr = cv2.imread('LabPhoto1.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)

# Pops up a window for displaying the CV_LOAD_IMAGE_GRAYSCALE
winname = 'imageWin'
win = cv.NamedWindow(winname, cv.CV_WINDOW_AUTOSIZE)

cv2.putText(im, 'motion', (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))
cv2.imshow('motion image', im)
cv2.waitKey(1000)
cv.DestroyWindow(winname)

invid = cv2.VideoCapture('LabVideo.MOV')
width = int(invid.get(cv.CV_CAP_PROP_FRAME_WIDTH))
height = int(invid.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
fps = int(invid.get(cv.CV_CAP_PROP_FPS))
length = int(invid.get(cv.CV_CAP_PROP_FRAME_COUNT))

for i in range(length):
  _, im = invid.read()
  if i % 3 == 0:
    cv2.imshow('fastForward',im)
    cv2.waitKey(100)

del invid
cv2.destroyAllWindows()
