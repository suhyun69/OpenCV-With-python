import cv2
import numpy as np

#1
# src에 512x512 크기의 3-채널 컬러 영상을 생성하고, cv2.rectangle()로 채워진 사각형을 그린다.
# cv2.cvtColor()로 그레이스케일 영상 gray로 변환한다
src = np.zeros(shape = (512, 512, 3), dtype=np.uint8)
cv2.rectangle(src, (50, 100), (450, 400), (255, 255, 255), -1)
cv2.rectangle(src, (100, 150), (400, 350), (0, 0, 0), -1)
cv2.rectangle(src, (200, 200), (300, 300), (255, 255, 255), -1)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

#2
# cv2.findContours()로 윤곽선 contours를 검출한다.
# mode = cv2.RETR_EXTERNAL로 리스트 contours에 len(contours) = 1개의 가장 외곽의 윤곽선을 검출한다.
# method = cv2.CHAIN_APPROX_SIMPLE로 윤곽선을 다각형으로 근사한 좌표를 반환한다.
# contours[0].shape = (4, 1, 2)은 4개의 검출된 좌표가 (1, 2) 배열에 저장된다.
# method = cv2.CHAIN_APPROX_NONE이면, contours[0].shape = (1400, 1, 2)로 윤곽선 위 모든 좌표 1400개를 검출한다
mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE
# mtehod = cv2.CHAIN_APPROX_NONE
contours, hierarchy = cv2.findContours(gray, mode, method)
print('type(contours)=', type(contours))
print('type(contours[0])=', type(contours[0]))
print('len(contours)=', len(contours))
print('type(contours[0].shape)=', type(contours[0].shape))
print('contours[0]=', contours[0])

#3
# cv2.drawContours(src, contours, -1, (255, 0, 0), 3)는 검출된 윤곽선 contours 전부를 src에 (255, 0, 0) 컬러로 두께 3으로 그린다
cv2.drawContours(src, contours, -1, (255, 0, 0), 3) # 모든 윤곽선

#4
# fopr 문으로 윤곽선의 (1, 2) 좌표 배열 pt에 의해서 중심점 (pt[0][0], pt[0][1]), 반직름 5인 (0, 0, 255) 컬러로 채워진 원을 src에 그린다.
for pt in contours[0][:]: # 윤곽선 좌표
    cv2.circle(src, (pt[0][0], pt[0][1]), 5, (0, 0, 255), -1)

cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()
