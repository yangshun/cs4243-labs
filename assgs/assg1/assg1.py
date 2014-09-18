# CS4243 Assignment 1
# ===================
# Name: Tay Yang Shun
# Matric: A0073063M

import sys
import cv2
import cv2.cv as cv
import numpy as np
import random

# Example usage: 
# python assg1.py airborne.jpg

IMAGE_FILE_NAME = sys.argv[1]
INTENSITY_RANGE = 256

image = cv2.imread(IMAGE_FILE_NAME, cv2.CV_LOAD_IMAGE_GRAYSCALE)
height, width = image.shape

flattened_image = image.reshape(1, -1)[0] # Convert to 1-d array

freq = [0] * 256
for px in flattened_image:
  freq[px] += 1
bin_size = sum(freq) // INTENSITY_RANGE

cum_freq = [0] * INTENSITY_RANGE
cum_freq[0] = freq[0]
for i in range(1, INTENSITY_RANGE):
  # Calculates cumulative frequency of each intensity
  cum_freq[i] = cum_freq[i-1] + freq[i]

freq_temp = []
for i in range(INTENSITY_RANGE):
  limit = bin_size * i
  index = 0
  for j, f in list(enumerate(cum_freq)):
    if limit <= f:
      freq_temp.append((i, j))
      break
freq_temp.append((None, 256))

freq_range_mapping = []
for i in range(INTENSITY_RANGE):
  if freq_temp[i][1] == freq_temp[i+1][1]:
    freq_range_mapping.append((i, (freq_temp[i][1], freq_temp[i][1])))
  else:
    freq_range_mapping.append((i, (freq_temp[i][1], freq_temp[i+1][1]-1)))

freq_map = {} # Dict that maps original intensity to a list of possible equalized intensities
for i in freq_range_mapping:
  new_intensity = i[0]
  for orig_intensity in range(i[1][0], i[1][1]+1):
    if orig_intensity in freq_map:
      freq_map[orig_intensity].append(new_intensity)
    else:
      freq_map[orig_intensity] = [new_intensity]

for px in np.nditer(image, op_flags=['readwrite']):
  # Iterate through image and replace with new intensity values
  new_value = random.sample(freq_map[int(px)], 1)[0]
  px[...] = new_value

file_name, file_extension = IMAGE_FILE_NAME.split('.')
new_file_name = file_name + '-equalized.' + file_extension
cv2.imwrite(new_file_name, image)
print 'Histogram equalized image \'' + new_file_name + '\' generated'
