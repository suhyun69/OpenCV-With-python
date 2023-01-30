import cv2
import numpy as np

src = cv2.imread('../data/rect.jpg', cv2.IMREAD_GRAYSCALE)

#1
# 입력 영상 src의 그래디어트 gx, gy를 cv2.Sobel() 함수를 계산한다
gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)

#2
# gx의 절대값의 제곱근을 계산하고, cv2.normalize()로 최소값을 0, 최대값을 255로 dstX에 정규화한다
dstX = cv2.sqrt(np.abs(gx))
dstX = cv2.normalize(dstX, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

#3
# gy의 절대값의 제곱근을 계산하고, cv2.normalize()로 최소값을 0, 최대값을 255로 dstY에 정규화한다
dstY = cv2.sqrt(np.abs(gy))
dstY = cv2.normalize(dstY, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

#4
# cv2.magnitude(gx, gy)로 그래디언트의 크기를 mag에 계산한다.
# mag의 값이 큰 화소가 에지이다.
# cv2.normalize()로 최소값을 0, 최대값을 255로 dstM에 정규화한다
mag = cv2.magnitude(gx, gy)
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(mag)
print('mag:', minVal, maxVal, minLoc, maxLoc)

dstM = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

cv2.imshow('src', src)
cv2.imshow('dstX', dstX)
cv2.imshow('dstY', dstY)
cv2.imshow('dstM', dstM)
cv2.waitKey()
cv2.destroyAllWindows()