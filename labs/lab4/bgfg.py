# CS4243 Lab 4
# ===================
# Name: Tay Yang Shun
# Matric: A0073063M

import os
import cv2
import cv2.cv as cv
import numpy as np

VIDEO_FILE_NAME = 'traffic.mov'

cap = cv2.VideoCapture(VIDEO_FILE_NAME)

frame_width = cap.get(cv.CV_CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv.CV_CAP_PROP_FPS)
frame_count = cap.get(cv.CV_CAP_PROP_FRAME_COUNT)
print 'width:', frame_width
print 'height:', frame_height
print 'frames per second:', fps
print 'frame count:', frame_count

frame_width = int(frame_width)
frame_height = int(frame_height)
fps = int(fps)
frame_count = int(frame_count)


_, img = cap.read()
avgImg = np.float32(img)

for fr in range(1, frame_count):
  _, img = cap.read()
  alpha = 1 / float(fr + 1)
  cv2.accumulateWeighted(img, avgImg, alpha)
  normImg = cv2.convertScaleAbs(avgImg) # convert into uint8 image 
  cv2.imshow('img', img)
  cv2.imshow('normImg', normImg)
  print 'fr = ', fr, 'alpha = ', alpha

cv2.waitKey(0) 
cv2.destroyAllWindows()

cap = cv2.VideoCapture(VIDEO_FILE_NAME) 
grAvgImg = cv2.cvtColor(normImg, cv2.COLOR_BGR2GRAY)

for fr in range(frame_count):
  _, img = cap.read()
  grImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  diffImg = cv2.absdiff(grImg, grAvgImg)
  thresh, biImg = cv2.threshold(diffImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

  # Extract foreground objects (with colour)
  fg = cv2.dilate(biImg, None, iterations=2)
  res = cv2.bitwise_and(img, img, fg, fg)

  bgtemp = cv2.erode(biImg, None, iterations=3)
  thresh2, bg = cv2.threshold(bgtemp, 2, 255, cv2.THRESH_BINARY_INV)

  cv2.imshow('Foreground Image', res)
  cv2.waitKey(10)
  
fg = res

cv2.imshow('Binarized Image', biImg)
cv2.imwrite('Binarized Image.jpg', biImg)
cv2.waitKey(0)

cv2.imshow('Foreground Image', fg)
cv2.imwrite('Foreground Image.jpg', fg)
cv2.waitKey(0)

cv2.imshow('Background image', bg)
cv2.imwrite('Background Image.jpg', bg)
