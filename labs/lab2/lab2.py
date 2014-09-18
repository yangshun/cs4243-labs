# CS4243 Lab 2
# ===================
# Name: Tay Yang Shun
# Matric: A0073063M

import cv2

ORIGINAL_FILE = 'scene.jpg'

RADIUS = 3
THICKNESS = -1
LINE_TYPE = 8
SHIFT = 0

image = cv2.imread(ORIGINAL_FILE, cv2.CV_LOAD_IMAGE_GRAYSCALE)
feature_list = cv2.goodFeaturesToTrack(image, 200, 0.001, 11.0).reshape((-1,2))

image = cv2.imread(ORIGINAL_FILE, cv2.CV_LOAD_IMAGE_COLOR)

for (x, y) in feature_list:
  cv2.circle(image, (x, y), RADIUS, (255, 255, 255, 0), THICKNESS, LINE_TYPE, SHIFT)

cv2.imwrite('corners.jpg', image)
