import cv2
import numpy as np

# 영상을 부드럽게 하는 블러링 blurring/스무딩 smoothing 필터를 사용하는 아래의 함수는 영상의 잡음 noise를 제거하고 영상을 부드럽게 한다
# boxFilter, bilateralFilter, medianBlur, blur, GaussianBlur, getGaussianKernel

src = cv2.imread('../data/lena.jpg', cv2.IMREAD_GRAYSCALE)

dst1 = cv2.medianBlur(src, ksize=7)
dst2 = cv2.blur(src, ksize=(7, 7))
dst3 = cv2.GaussianBlur(src, ksize=(7,7), sigmaX=0.0)
dst3 = cv2.GaussianBlur(src, ksize=(7,7), sigmaX=10.0)

cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.imshow('dst3', dst3)
cv2.waitKey()
cv2.destroyAllWindows()

