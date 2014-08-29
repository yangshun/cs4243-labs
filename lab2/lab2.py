import cv2

image = cv2.imread('scene.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)

feature_list = cv2.goodFeaturesToTrack(image, 200, 0.001, 11.0)

image = cv2.imread('scene.jpg', cv2.CV_LOAD_IMAGE_COLOR)

width = image.shape[1] 
height = image.shape[0]

feature_list = feature_list.reshape((-1,2))

for (x, y) in feature_list:
  center = (x, y)
  radius = 3
  thickness = -1 # negative means filled
  lineType = 8
  shift = 0
  cv2.circle(image, center, radius, (0, 0, 255, 0), thickness, lineType, shift)

cv2.imwrite('features.jpg', image)
