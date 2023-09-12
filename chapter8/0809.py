import cv2
import numpy as np

#1
# 입력 영상 src를 그레이스케일 gray로 변환하고 이진 영상 bImage를 생성한다
src = cv2.imread('../data/momentTest.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, bImage = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

#2
# cv2.moments()로 이진 영상 bImage에서 영상 모멘트 M을 계산한다
# cv2.moments(bImage)는 bImmage의 물체 영역의 화소값 255로 계산하고, cv2.moments(bImage, True)는 1로 계산하여 모멘트 값이 다르다.
# for 문으로 M.items(0의 key, value 값으로 모멘트를 출력한다
M = cv2.moments(bImage, True)
for key, value in M.items():
    print('{}={}'.format(key, value))

#3
# 물체의 중심좌표를 cx, cy로 계산하고 dst에 빨간색 원으로 표시한다
cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])
dst = src.copy()
cv2.circle(dst, (cx, cy), 5, (0, 0, 255), 2)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()