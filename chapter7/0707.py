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
# mode = cv2.RETR_LIST로 리스트 contours에 len(contours) = 3개의 모든 윤곽선을 검출한다.
# method = cv2.CHAIN_APPROX_SIMPLE로 윤곽선을 다각형으로 근사한 좌표를 반환한다.
# 윤곽선 contours[0]은 contours[0].shape = (4, 1, 2)로 4개의 검출된 좌표가 (1, 2) 배열에 저장된다.
# cv2.drawContours(src, contours, -1, (255, 0, 0), 3)는 리스트 contours의 모든 윤곽선을 그린다.
mode = cv2.RETR_LIST
method = cv2.CHAIN_APPROX_SIMPLE
contours, hierarchy = cv2.findContours(gray, mode, method)

print('len(contours)=', len(contours))
print('contours[0].shape=', contours[0].shape)
print('contours[0]=', contours[0])

#3
# fopr 문으로 리스트 contours의 각 윤곽선 cnt를 cv2.drawContours(src, [cnt], 0, (255, 0, 0), 3)로 그리고
# 그리고 for 문으로 cnt의 각 좌표 pt가 중심인 반지름 5의 원을 (0, 0, 255) 컬러로 그린다.
for cnt in contours:
    cv2.drawContours(src, [cnt], 0, (255, 0, 0), 3)

    for pt in cnt:
        cv2.circle(src, (pt[0][0], pt[0][1]), 5, (0, 0, 255), -1)

cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()