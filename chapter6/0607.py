import cv2
import numpy as np

src = cv2.imread('../data/rect.jpg', cv2.IMREAD_GRAYSCALE)

#1
# cv2.getDerivKernels()에서 dx=1, dy=0, ksize=3으로 x-축 방향 미분 선형 필터 kx, ky를 생성한다.
# sobelX 필터는 dx=1, dy=0에서의 Sobel에서 필터이다.
# cv2.filter2D()로 src에 sobelX 필터를 적용하여 gx를 생성한다.
kx, ky = cv2.getDerivKernels(1, 0, ksize=3)
sobelX = ky.dot(kx.T)
print('kx=', kx)
print('ky=', ky)
print('sobleX=', sobelX)
gx = cv2.filter2D(src, cv2.CV_32F, sobelX)

#2
# cv2.getDerivKernels()에서 dx=0, dy=1, ksize=3으로 y-축 방향 미분 선형 필터 kx, ky를 생성한다.
# sobelY 필터는 dx=0, dy=1에서의 Sobel에서 필터이다.
# cv2.filter2D()로 src에 sobelY 필터를 적용하여 gy를 생성한다.
kx, ky = cv2.getDerivKernels(0, 1, ksize=3)
sobelY = ky.dot(kx.T)
print('kx=', kx)
print('ky=', ky)
print('sobleY=', sobelY)
gy = cv2.filter2D(src, cv2.CV_32F, sobelY)

#3
# cv2.magnitude()로 그래디언트(gx, gy)의 크기를 mag에 계산하고, 임계값 100을 사용하여 에지 이진 영상 edge를 생성한다.
mag = cv2.magnitude(gx, gy)
ret, edge = cv2.threshold(mag, 100, 255, cv2.THRESH_BINARY)

cv2.imshow('edge', edge)
cv2.waitKey()
cv2.destroyAllWindows()
