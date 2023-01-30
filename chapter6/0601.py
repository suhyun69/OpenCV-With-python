import cv2
import numpy as np

# 영상을 부드럽게 하는 블러링 blurring/스무딩 smoothing 필터를 사용하는 아래의 함수는 영상의 잡음 noise를 제거하고 영상을 부드럽게 한다
# boxFilter, bilateralFilter, medianBlur, blur, GaussianBlur, getGaussianKernel

src = cv2.imread('../data/lena.jpg', cv2.IMREAD_GRAYSCALE)

dst1 = cv2.boxFilter(src, ddepth=-1, ksize=(11, 11))
dst2 = cv2.boxFilter(src, ddepth=-1, ksize=(21, 21))

# bilateralFilter 함수는 가우시안 함수를 사용하여 에지를 덜 약화하면서 양방향 필터링을 한다.
dst3 = cv2.bilateralFilter(src, d=11, sigmaColor=10, sigmaSpace=10)
dst4 = cv2.bilateralFilter(src, d=-1, sigmaColor=10, sigmaSpace=10)

cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.imshow('dst3', dst3)
cv2.imshow('dst4', dst4)
cv2.waitKey()
cv2.destroyAllWindows()

