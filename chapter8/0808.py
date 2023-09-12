import cv2
import numpy as np

#1
# src 영상은 6x4의 검은색 원 패턴을 갖는다.
# src 또는 gray에서 cv2.findCirclesGrid()로 patternSize = (6, 4)의 원의 중심점을 centers에 검출한다
# found = True이고, centers.shape = (24, 1, 2)으로 24개의 중심점의 좌표를 (1, 2)에 저장한다
src = cv2.imread('../data/circleGrid.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
patternSize = (6, 4)
found, centers = cv2.findCirclesGrid(src, patternSize)
print('centers.shape=', centers.shape)

#2
# cv2.cornerSubPix()로 centers를 부화소 수준으로 centers2에 계산한다
term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)
centers2 = cv2.cornerSubPix(gray, centers, (5, 5), (-1, -1), term_crit)

#3
# src를 dst에 복사하고, cv2.drawChessboardCorners()로 검출된 centers2를 dst에 그려 표시한다
dst = src.copy()
cv2.drawChessboardCorners(dst, patternSize, centers2, found)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()