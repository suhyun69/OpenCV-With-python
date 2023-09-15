import cv2
import numpy as np

#1
# HSV 영상 hsv에서 cv2.inRange로 손 영역을 검출한 이진 영상 bImage를 생성한다.
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
# cv2.convexHull()로 cnt에서 returnPOints = False를 적용하여 볼록 껍질 hull을 계산한다. 
# hull은 cnt의 첨자를 이용하여 hull_points = cnt[hull[:,0]]로 hull_points 좌표에 볼록 껍질의 좌표를 hull_points에 계산하고, dst2에 (255, 0, 255) 색상과 두께 6으로 그린다
dst2 = dst.copy()
rows, cols = dst2.shape[:2]
hull = cv2.convexHull(cnt, returnPoints = False)
hull_points = cnt[hull[:,0]]
cv2.drawContours(dst2, [hull_points], 0, (255, 0, 255), 6)

#3
# cv2.convexityDefects()로 cnt에서 계산한 hull을 이용하여 볼록 결함 defects를 계산한다
# defects.shape = (24, 1, 4)이다.
# for 문에서 각 볼록 결함에 대해 s, e, f, d = defects[i, 0]로 저장한다
# s, e, f는 cnt의 첨자이다. dist = d / 256로 거리를 계산하고, s, e, f 첨자를 이용하여 start, end, far 좌표를 계산한다
# dist > T이면 직선과 원으로 dist2에 표시한다
# T가 작을수록 더 많은 볼록 결함이 발견된다
T = 5
defects = cv2.convexityDefects(cnt, hull)
print('defects.shape=', defects.shape)
for i in range(defects.shape[0]):
    s, e, f, d = defects[i, 0]
    dist = d / 256
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    if dist > T:
        cv2.line(dst2, start, end, [255, 255, 0], 2)
        cv2.line(dst2, start, far, [0, 255, 0], 1)
        cv2.line(dst2, end, far, [0, 255, 0], 1)

        cv2.circle(dst2, start, 5, [0, 255, 255], -1)
        cv2.circle(dst2, end, 5, [0, 128, 255], -1)
        cv2.circle(dst2, far, 5, [0, 0, 255], -1)
cv2.imshow('dst2', dst2)

cv2.waitKey()
cv2.destroyAllWindows()

