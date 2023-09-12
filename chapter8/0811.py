import cv2
import numpy as np

#1
# 입력 영상 src를 그레이스케일 gray로 변환하고, 이진 영상 bImage를 구하고, cv2.findContours()로 bImage에서 윤곽선(경계선)을 contours에 검출하고,
# 첫 번째 윤곽선 contours[0]을 cnt에 저장하고 src를 복사한 dst에 파란색으로 윤곽선을 그린다
src = cv2.imread('../data/momentTest.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, bImage = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE
contours, hierarchy = cv2.findContours(bImage, mode, method)

dst = src.copy()
cnt = contours[0]
cv2.drawContours(dst, [cnt], 0, (255, 0, 0), 3)

#2
# cv2.moments()로 윤곽선 cnt의 경계선 모멘트 M을 계산하고 cv.HuMoments()로 모멘트를 이용하여 Hu의 모멘트를 hu에 계산한다
M = cv2.moments(cnt)
hu = cv2.HuMoments(M)
print('hu.shape=', hu.shape)
print('hu=', hu)

#3
# cv2.getRotationMatrix2D()로 center를 중심으로 angle=45도 회전하고, scale=0.2로 축소하는 2x3 어파인 변환행렬 A를 계산하고, A[:, 2] += t로 어파인 변환행렬 A에 t만큼의 이동을 반영한다
# cv2.transform()로 윤곽선 cnt에 어파인 행렬 A를 적용하여 변환 윤곽선 cnt2를 생성하고, dst에 초록색으로 표시한다
angle = 45.0
scale = 0.2
cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])
center = (cx, cy)
t = (20, 30)
A = cv2.getRotationMatrix2D(center, angle, scale)
A[:, 2] += t # translation
print('A=', A) # Affine 변환
cnt2 = cv2.transform(cnt, A)
cv2.drawContours(dst, [cnt2], 0, (0, 255, 0), 3)
cv2.imshow('dst', dst)

#4
# cv2.moments()로 윤곽선 cnt2의 경계선 모멘트를 M2에 계산하고, cv2.HuMoments()로 모멘트를 이용하여 Hu의 모멘트를 hu2에 계산한다
M2 = cv2.moments(cnt2)
hu2 = cv2.HuMoments(M2)
print('hu2.shape=', hu2.shape)
print('hu2=', hu)

#5
# hu, hu2의 차이의 절대값 합계를 계산하면 diffSum은 매우 작은 값을 갖는다. 즉, hu의 모멘트가 어파인 변환에 불변인 것을 의미한다.
# hu와 hu2가 정확히 갖지 않은 것은 영상의 해상도 문제이다
diffSum = sum(abs(hu - hu2))
print('diffSum=', diffSum)

cv2.waitKey()
cv2.destroyAllWindows()