import cv2
import numpy as np
src = cv2.imread('../data/srcThreshold.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('src', src) # src는 전체적으로 조명이 일정하지 않다

ret, dst = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('dst', dst)

dst2 = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 7) # 각각의 화소별로 적응형 임계값을 적용한 영상
cv2.imshow('dst2', dst2)

dst3 = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 7) # 각각의 화소별로 적응형 임계값을 적용한 영상
cv2.imshow('dst3', dst3)

cv2.waitKey()
cv2.destroyAllWindows()