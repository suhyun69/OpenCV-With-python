import cv2
import numpy as np

src = cv2.imread('../data/lena.jpg', cv2.IMREAD_GRAYSCALE)

minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(src) # cv2.minMaxLoc(src)은 src의 최소값, 최대값, 최소값 위치, 최대값 위치를 계산하여 반환한다
print('src:', minVal, maxVal, minLoc, maxLoc)

dst = cv2.normalize(src, None, 100, 200, cv2.NORM_MINMAX) # norm_type = cv2.NORM_MINMAX에 의해, src의 최소/최소값 범위[18.0, 248.0]을 범위 [100, 200]으로 dst에 정규화한다. dst = None은 결과 영상을 새로 생성한다
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dst)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()