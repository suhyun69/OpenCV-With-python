import cv2
import numpy as np

#1
# 'hand.jpg' 영상을 3-채널 BGR 컬러 영상으로 읽은 src1을 HSV 영상으로 hsv1에 변환하고, hsv1에서 cv2.inRange()로 lowerb1 = (0, 40, 0), upperb1 = (20, 180, 255) 범위를 적용하여 손 영역을 분할한다
src1 = cv2.imread('../data/hand.jpg')
hsv1 = cv2.cvtColor(src1, cv2.COLOR_BGR2HSV)
lowerb1 = (0, 40, 0)
upperb1 = (20, 180, 255)
dst1 = cv2.inRange(hsv1, lowerb1, upperb1)

#2
# 'flower.jpg' 영상을 3-채널 BGR 컬러 영상으로 읽은 src2을 HSV 영상으로 hsv2에 변환하고, hsv2에서 cv2.inRange()로 lowerb2 = (150, 100, 100), upperb2 = (180, 255, 255 범위를 적용하여 꽃 영역을 분할한다
src2 = cv2.imread('../data/flower.jpg')
hsv2 = cv2.cvtColor(src2, cv2.COLOR_BGR2HSV)
lowerb2 = (150, 100, 100)
upperb2 = (180, 255, 255)
dst2 = cv2.inRange(hsv2, lowerb2, upperb2)

#3
cv2.imshow('src1', src1)
cv2.imshow('dst1', dst1)
cv2.imshow('src2', src2)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()
