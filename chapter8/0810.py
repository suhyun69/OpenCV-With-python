import cv2
import numpy as np

#1
# 입력 영상 src를 그레이스케일로 gray로 변환하고, cv2.THRESH_BINARY_INV로 물체가 흰색, 배경이 검은색인 이진 영상 bImage를 생성한다
src = cv2.imread('../data/circles.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, bImage = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

#2
# cv2.findContours()로 이진 영상 bImage에서 윤곽선(경계선)을 contours에 검출하고, dst에 파란색으로 윤곽선을 그린다
mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE
contours, hierarchy = cv2.findContours(bImage, mode, method)

dst = src.copy()
cv2.drawContours(dst, contours, -1, (255, 0, 0), 3) # 모든 윤곽선

#3
# contours의 각 윤곽선 cnt의 모멘트를 M에 계산하고, 각 윤곽선의 중심좌표 (cx, cy)를 계산하고, dst에 빨간색 원으로 표시한다
for cnt in contours:

    M = cv2.moments(cnt, True)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cv2.circle(dst, (cx, cy), 5, (0, 0, 255), 2)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()