import cv2
import numpy as np

# shape = cv2.MORPH_RECT, ksize=(3,3)로 사각형 kernel을 생성하고, cv2.morphologyEx() 함수로 모폴로지 연산을 수행한다.
src = cv2.imread('../data/morphology.jpg', cv2.IMREAD_GRAYSCALE)
kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3,3))

# 입력 영상 src에 cv2.MORPH_CLOSE 연산을 5회 수행하여 흰색 물체 속의 검은색 잡음을 제거
closing = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel, iterations = 5)

# 입력 영상 src에 cv2.MORPH_OPEN 연산을 5회 수행하여 흰색 잡음을 제거
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations = 5)

# opening에 cv2.MORPH_GRADIENT 연산을 1회 수행하여 흰색 물체의 테두리를 검출
gradient = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel)

# 입력 영상 src에 cv2.MORPH_TOPHAT 연산을 5회 수행하여 검은색 배경 속의 흰색 점을 검출
tophat = cv2.morphologyEx(src, cv2.MORPH_TOPHAT, kernel, iterations = 5)

# 입력 영상 src에 cv2.MORPH_BLACKHAT 연산을 5회 수행하여 흰색 물체 속의 검은색 점을 검출
blackhat = cv2.morphologyEx(src, cv2.MORPH_BLACKHAT, kernel, iterations = 5)

cv2.imshow('opening', opening)
cv2.imshow('closing', closing)
cv2.imshow('gradient', gradient)
cv2.imshow('tophat', tophat)
cv2.imshow('blackhat', blackhat)
cv2.waitKey()
cv2.destroyAllWindows()