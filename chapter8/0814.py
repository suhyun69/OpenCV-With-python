import cv2
import numpy as np

#1
# HSV 영상 hsv에서 cv2.inRange()로 손 영역을 검출한 이진 영상 bimage를 생성한다
# cv2.findContours()로 bImage에서 윤곽선을 contours에 검출하고 contours[0]를 cnt에 저장하고, src를 복사한 dst에 파란색 윤곽선으로 표시한다
src = cv2.imread('../data/hand.jpg')
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
lowerb = (0, 40, 0)
upperb = (20, 180, 255)
bImage = cv2.inRange(hsv, lowerb, upperb)

mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE
contours, hierarchy = cv2.findContours(bImage, mode, method)

dst = src.copy()
cnt = contours[0]
cv2.drawContours(dst, [cnt], 0, (255, 0, 0), 2)

#2
# cv2.convexHull()로 cnt에서 볼록 껍질 hull을 계산한다. dst를 복사한 dst2에 빨간색 직선으로 볼록 껍질을 그린다
dst2 = dst.copy()
rows, cols = dst2.shape[:2]
hull = cv2.convexHull(cnt)
cv2.drawContours(dst2, [hull], 0, (0, 0, 255), 2)
cv2.imshow('dst2', dst2)

cv2.waitKey()
cv2.destroyAllWindows()