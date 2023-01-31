import cv2
import numpy as np

src = cv2.imread('../data/lena.jpg', cv2.IMREAD_GRAYSCALE)

# 입력 영상 src에서 threshold1 = 50, threshold2 = 100, apertureSize = 3으로 Canny 에지를 edges1에 검출한다.
# 검출된 에지가 Sobel 에지보다 가늘어진 것을 볼 수 있다
edges1 = cv2.Canny(src, 50, 100)

# 입력 영상 src에서 threshold1 = 50, threshold2 = 200, apertureSize = 3으로 Canny 에지를 edges2에 검출한다.
# 검출된 에지 edges2는 edges1보다 적은 에지가 검출된다.
edges2 = cv2.Canny(src, 50, 200)

cv2.imshow('edge1', edges1)
cv2.imshow('edge2', edges2)
cv2.waitKey()
cv2.destroyAllWindows()