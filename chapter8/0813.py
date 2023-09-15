import cv2
import numpy as np

#1
# 입력 영상 src를 그레이스케일로 gray로 변환하고, 이진 영상 bImage를 생성한다. cv2.findContours()로 bImage에서 윤곽선을 contours로 검출하고 contours[0]를 cnt에 저장하여 src를 복사한 dst에 파란색으로 표시한다
src = cv2.imread('../data/banana.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, bImage = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
bImage = cv2.dilate(bImage, None)

mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE
contours, hierarchy = cv2.findContours(bImage, mode, method)

dst = src.copy()
cnt = contours[0]
cv2.drawContours(dst, [cnt], 0, (255, 0, 0), 3)

#2
# cv2.fitLine()로 cnt를 직선 [vx, vy, x, y]로 근사한다. x1 = 0, x2 = col-1에서의 y 좌표 y1, y2를 계산하여 dst를 복사한 dst2에 빨간색 직선을 그린다
dst2 = dst.copy()
rows, cols = dst2.shape[:2]
[vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
y1 = int((-x * vy / vx) + y)
y2 = int(((cols - x) * vy / vx) + y)
cv2.line(dst2, (0, y1), (cols - 1, y2), (0, 0, 255), 2)
cv2.imshow('dst2', dst2)

#3
# cv2.fitEllipse()로 cnt를 타원 ellipse로 근사하고, dst를 복사한 dst3에 빨간색 타원을 그린다
ellipse = cv2.fitEllipse(cnt)
dst3 = dst.copy()
cv2.ellipse(dst3, ellipse, (0, 0, 255), 2)
cv2.imshow('dst3', dst3)

#4
# cv2.approxPolyDP()로 cnt를 epsilon = 20, closed = True를 적용하여 다각형 poly로 근사하고, dst를 복사한 dst4에 빨간색 다각형을 그린다
poly = cv2.approxPolyDP(cnt, epsilon = 20, closed = True)
dst4 = dst.copy()
cv2.drawContours(dst4, [poly], 0, (0, 0, 255), 2)
cv2.imshow('dst4', dst4)

#5
# cv2.pointPolygonTest()로 (x,y)가 cnt의 내부 점일 때 dst5[x, y]를 초록색으로 변경하면 바나나의 내부 화소가 변경된다
dst5 = dst.copy()
for y in range(rows):
    for x in range(cols):
        if cv2.pointPolygonTest(cnt, (x, y), False) > 0:
            dst5[y, x] = (0, 255, 0)
cv2.imshow('dst5', dst5)

cv2.waitKey()
cv2.destroyAllWindows()