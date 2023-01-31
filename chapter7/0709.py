import cv2
import numpy as np

#1
# src에 512x512 크기의 그레이스케일 영상을 생성하고, cv2.rectangle()로 채워진 사각형을 그린다.
src = np.zeros(shape = (512, 512), dtype = np.uint8)
cv2.rectangle(src, (50, 200), (450, 300), (255, 255, 255), -1)

#2
# src에 cv2.distanceTransform()로 distanceType = cv2.DIST_L1, maskSize = 3를 적용하여 dist에 거리를 계산한다.
# cv2.minMaxLoc(dist)로 계산한 최대값은 maxVal = 51.0이다.
# cv2.normalize()로 dist를 [0, 255] qjadnlfh 정규화한다.
# cv2.threshold()로 dist를 thresh = maxVal-1로 dst2에 임계값을 적용한다.
dist = cv2.distanceTransform(src, distanceType = cv2.DIST_L1, maskSize = 3)
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dist)
print('src:', minVal, maxVal, minLoc, maxLoc)

dst = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
ret, dst2 = cv2.threshold(dist, maxVal-1, 255, cv2.THRESH_BINARY)

#3
# cv2.Sobel()로 거리 dist에서 그래디언트를 계산하고, 크기 mag를 계산하여 thresh = maxVal - 2, cv2.THRESH_BINARY_INV로 임계값 영상을 생성한다.
gx = cv2.Sobel(dist, cv2.CV_32F, 1, 0, ksize = 3)
gy = cv2.Sobel(dist, cv2.CV_32F, 0, 1, ksize = 3)
mag = cv2.magnitude(gx, gy)
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(mag)
print('src:', minVal, maxVal, minLoc, maxLoc)
ret, dst3 = cv2.threshold(mag, maxVal - 2, 255, cv2.THRESH_BINARY_INV)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.imshow('dst2', dst2)
cv2.imshow('dst3', dst3)
cv2.waitKey()
cv2.destroyAllWindows()
