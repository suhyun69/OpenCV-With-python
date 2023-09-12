import cv2
import numpy as np

#1
# src 영상은 7x4의 흰색과 검은색 사각형을 갖는다. 
# 체스보드 패턴의 내부 코너점은 6x3이다. 
# src 또는 gray에서 cv2.findCheckssboardCorners()로 patternSize = (6, 3)의 패턴 크기의 코너점을 corners에 검출한다.
# found = True이고, corners.shape = (18, 1, 2)으로 18개의 코너점의 좌표를 (1, 2)에 저장한다
src = cv2.imread('../data/chessBoard.jpg')
gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
patternSize = (6, 3)
found, corners = cv2.findChessboardCorners(src, patternSize)
print('corners.shape=', corners.shape)

#2
# cv2.cornerSubPix()로 corners를 부화소 수준으로 corners2에 계산한다
term_crit = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)
corners2 = cv2.cornerSubPix(gray, corners, (5,5), (-1, -1), term_crit)

#3
# src를 dst에 복사하고, cv2.drawChessboardCorners()로 검출된 코너점 corners2를 dst에 그려 표시한다
dst = src.copy()
cv2.drawChessboardCorners(dst, patternSize, corners2, found)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()