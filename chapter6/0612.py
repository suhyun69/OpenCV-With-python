import cv2
import numpy as np

src = cv2.imread('../data/alphabet.bmp', cv2.IMREAD_GRAYSCALE)
tmp_A = cv2.imread('../data/A.bmp', cv2.IMREAD_GRAYSCALE)
tmp_S = cv2.imread('../data/S.bmp', cv2.IMREAD_GRAYSCALE)
tmp_b = cv2.imread('../data/b.bmp', cv2.IMREAD_GRAYSCALE)
dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

#1
# 참조 영상 src에서 템플릿 tmp_A를 cv2.TM_SQDIFF_NORMED 방법으로 매칭한 결과를 R1에 저장한다.
# cv2.minMaxLoc(R1)로 최소값 minVal와 최소값 위치 minLoc을 찾는다.
# cv2.rectangle()로 최소값 위치 minLoc와 tmp_A의 크기(h, w)를 이용한 모서리 좌표(minLoc[0] + w, minLoc[1] + h로 dst에 사각형을 표시한다.
R1 = cv2.matchTemplate(src, tmp_A, cv2.TM_SQDIFF_NORMED)
minVal, _, minLoc, _ = cv2.minMaxLoc(R1)
print('TM_SQDIFF_NORMED:', minVal, minLoc)

h, w = tmp_A.shape[:2]
cv2.rectangle(dst, minLoc, (minLoc[0] + w, minLoc[1] + h), (255, 0, 0), 2)

#2
# 참조 영상 src에서 템플릿 tmp_S를 cv2.TM_CCORR_NORMED 방법으로 매칭한 결과를 R2에 저장한다.
# cv2.minMaxLoc(R2)로 최대값 maxVal와 최대값 위치 maxLoc을 찾는다.
# cv2.rectangle()로 최대값 위치 maxLoc을 tmp_ㄴ의 크기(h, w)를 이용한 모서리 좌표(maxLoc을[0] + w, maxLoc을[1] + h로 dst에 사각형을 표시한다.
R2 = cv2.matchTemplate(src, tmp_S, cv2.TM_CCORR_NORMED)
_, maxVal, _, maxLoc = cv2.minMaxLoc(R2)
print('TM_CCORR_NORMED:', maxVal, maxLoc)
h, w = tmp_S.shape[:2]
cv2.rectangle(dst, maxLoc, (maxLoc[0] + w, maxLoc[1] + h), (0, 255, 0), 2)

#3
# 참조 영상 src에서 템플릿 tmp_b를 cv2.TM_CCOEFF_NORMED 방법으로 매칭한 결과를 R2에 저장한다.
# cv2.minMaxLoc(R3)로 최대값 maxVal와 최대값 위치 maxLoc을 찾는다.
# cv2.rectangle()로 최대값 위치 maxLoc을 tmp_ㄴ의 크기(h, w)를 이용한 모서리 좌표(maxLoc을[0] + w, maxLoc을[1] + h로 dst에 사각형을 표시한다.
R3 = cv2.matchTemplate(src, tmp_b, cv2.TM_CCOEFF_NORMED)
_, maxVal, _, maxLoc = cv2.minMaxLoc(R3)
print('TM_CCOEFF_NORMED:', maxVal, maxLoc)
h, w = tmp_b.shape[:2]
cv2.rectangle(dst, maxLoc, (maxLoc[0] + w, maxLoc[1] + h), (0, 0, 255), 2)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()