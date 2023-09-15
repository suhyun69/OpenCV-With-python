import cv2
import numpy as np

#1
# 입력 영상 src를 그레이스케일로 gray로 변환하고, cv2.THRESH_BINARY_INV로 이진 영상 bImage를 구한 뒤에 cv2.dilate()로 흰색 바나나 내부의 검은색 점을 채워 bImage를 생성한다
src = cv2.imread('../data/banana.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, bImage = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
bImage = cv2.dilate(bImage, None)
cv2.imshow('bImage', bImage)

#2
# cv2.findContours()로 bImage에서 윤곽선을 contours에 검출하고, cv2.arcLength()로 윤곽선의 길이를 계산한 뒤에 길이가 가장 큰 윤곽선을 cnt에 계산하여 src를 복사한다 dst2에 파란색 윤곽선으로 표시한다
mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE
contours, hierarchy = cv2.findContours(bImage, mode, method)
print('len(contours)=', len(contours))

maxLength = 0
k = 0
for i, cnt in enumerate(contours):
    perimeter = cv2.arcLength(cnt, closed = True)
    if perimeter > maxLength:
        maxLength = perimeter
        k = i
print('maxLength=', maxLength)
cnt = contours[k]
dst2 = src.copy()
cv2.drawContours(dst2, [cnt], 0, (255, 0, 0), 3)

#3
# cv2.contourArea()로 cnt의 내부 면적을 area에 계산한다. cv2.boundingRect()로 cnt의 바운딩 사각형을 x, y, width, height에 계산하여 dst2를 복사한 dst3에 빨간색 사각형으로 표시한다
area = cv2.contourArea(cnt)
print('area=', area)
x, y, width, height = cv2.boundingRect(cnt)
dst3 = dst2.copy()
cv2.rectangle(dst3, (x, y), (x + width, y + height), (0, 0, 255), 2)
cv2.imshow('dst3', dst3)

#4
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int32(box)
print('box=', box)
dst4 = dst2.copy()
cv2.drawContours(dst4, [box], 0, (0, 0, 255), 2)
cv2.imshow('dst4', dst4)

#5
# cv2.minEnclosingCircle()로 최소 면적 원을 (x, y), radius에 찾고, dst2를 복사한 dst5에 빨간색 원으로 표시한다
(x, y), radius = cv2.minEnclosingCircle(cnt)
dst5 = dst2.copy()
cv2.circle(dst5, (int(x), int(y)), int(radius), (0, 0, 255), 2)
cv2.imshow('dst5', dst5)

cv2.waitKey()
cv2.destroyAllWindows()