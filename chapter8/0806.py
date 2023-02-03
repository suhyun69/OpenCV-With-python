import cv2
import numpy as np

#1
# cv2.goodFeaturesToTrack()로 gray에서 최대 코너점 maxCorners=K을 적용하여 코너점을 corners에 검출한다.
# K=5인 경우, corners.shape(5,1,2)로 5개의 코너점의 좌표가 (1,2)에 저장된다.
src = cv2.imread('../data/CornerTest.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

K = 5
# K = 10
corners = cv2.goodFeaturesToTrack(gray, maxCorners=K, qualityLevel=0.05, minDistance=10)
print('corners.shape=', corners.shape)
print('corners=', corners)

#2
# cv2.goodFeaturesToTrack()로 gray에서 최대 코너점 maxCorners=K, useHarrisDetector=True를 적용하여 코너점을 corners2에 검출한다.
# K = 5인 경우, corners2.shape(5,1,2)로 5개의 코너점의 좌표가 (1,2)에 저장된다.
corners2 = cv2.goodFeaturesToTrack(gray, maxCorners=K, qualityLevel=0.05, minDistance=10, useHarrisDetector=True, k=0.04)
print('corners2.shape=', corners.shape)
print('corners2=', corners)

#3
# corners의 각 좌표에 cv2.circle()로 반지름 5인 빨간색 채워진 원으로 dst에 표시한다.
# corners2의 각 좌표에 cv2.circle()로 반지름 5인 파란색, 두께 2인 원으로 dst를 표시한다.
dst = src.copy()
corners = corners.reshape(-1, 2)
for x, y, in corners:
    cv2.circle(dst, (int(x), int(y)), 5, (0,0,255), -1)

corners2 = corners2.reshape(-1, 2)
for x, y, in corners2:
    cv2.circle(dst, (int(x), int(y)), 5, (255, 0, 0), -1)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()