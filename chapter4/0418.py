import cv2
import numpy as np

src1 = cv2.imread('../data/lena.jpg')
src2 = cv2.imread('../data/opencv_logo.png')
cv2.imshow('src2', src2)

#1
rows, cols, channels = src2.shape
roi = src1[0:rows, 0:cols] # 전체 크기에 대한 src1의 영역을 roi에 저장

#2
gray = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY) # 컬러 영상 src2를 그레이스케일 영상 gray로 변환하고,
ret, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY) # 전경과 배경을 분할하기 위해 이진 영상 mask를 생성하고
mask_inv = cv2.bitwise_not(mask) # 비트 반전 영상 mask_inv를 생성한다
cv2.imshow('mask', mask)
cv2.imshow('mask_inv', mask_inv)

#3
src1_bg = cv2.bitwise_and(roi, roi, mask=mask) # roi 영상에서 mask의 255(흰색영역)인 화소에서만 bitwise_and() 함수로 src1의 배경 영역을 복사하고, 전경영역은 0(검은색)으로. // Background
cv2.imshow('src1_bg', src1_bg)

#4
src2_fg = cv2.bitwise_and(src2, src2, mask=mask_inv) # cv2.bitwise_and()로 mask_inv 마스크를 사용하여 src2에서 전경영역을 src2_fg에 복사한다 // Foreground
cv2.imshow('src2_fg', src2_fg)

#5
dst = cv2.bitwise_or(src1_bg, src2_fg) # cv2.bitwise_or()로 src_bg와 src_fg를 비트 OR 연산하여 dst를 생성한다. cv2.add() 함수를 사용해도 결과는 같다
cv2.imshow('dst', dst)

#6
src1[0:rows, 0:cols] = dst

cv2.imshow('result', src1)
cv2.waitKey(0)
cv2.destroyAllWindows()

