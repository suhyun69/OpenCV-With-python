import cv2
import numpy as np

#1
# 입력 영상 src를 그레이스케일 영상 gray로 변환하고, 임계치를 이용하여 이진 영성 bimage를 생성한다.
# cv2.distanceTransform()로 bImage에서 거리 배열 dist를 계산한다.
# 거리를 보여주기 위해 8비트 영상으로 dist8에 변환한다.
src = cv2.imread('../data/circles2.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, bImage = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
dist = cv2.distanceTransform(bImage, cv2.DIST_L1, 3)
dist8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imshow('bImage', bImage)
cv2.imshow('dist8', dist8)

#2
# mask = (dist > maxVal * 0.5)로 거리 dst에서 최대값 maxVal을 이용하여 8비트 mask 영상을 계산한다.
# dist 대신 dist8을 이용할 수 있다.
# 여기서 중요한 점은 이진 영상 bImage에서 겹쳐진 원이 거리 계산을 이용한 mask에서는 분리된 것을 알 수 있다.
# 이렇게 분리하기 위하여 cv2.distanceTransform()을 이용한 것이다.
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dist)
print('dist:', minVal, maxVal, minLoc, maxLoc)
mask = (dist> maxVal * 0.5).astype(np.uint8) * 255
cv2.imshow('mask', mask)

#3
# cv2.findContours()로 mask에서 윤곽선 contours를 검출한다.
# 윤곽선 conturs[i]를 markers에 i+1로 채워 마커를 생성하고, cv2.watershed()로 src에서 markers에 표시된 마커 정보를 이용하여 영역을 markers에 분할한다.
mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE
contours, hierarchy = cv2.findContours(mask, mode, method)
print('len(contours)=', len(contours))

markers = np.zeros(shape=src.shape[:2], dtype=np.int32)
for i, cnt in enumerate(contours):
    cv2.drawContours(markers, [cnt], 0, i+1, -1)

#4
# src를 dst에 복사하고, dst[markers == -1] = [0, 0, 255]에 의해 markers에 -1인 경계선을 빨간색 [0,0, 255]로 변경한다
# for문에서 r,g,b에 [0, 255] 사이의 난수를 생성하여 dst[markers == i+1] = [b,g,r]로 markers == i+1인 dst의 화소를 [b,g,r] 컬러로 변경한다.
# cv2.addWeighted()로 src() * 0.4와 dst * 0.6으로 섞어 dst에 저장하고, 'dst' 윈도우에 표시한다.
dst = src.copy()
cv2.watershed(src, markers)

dst[markers == -1] = [0, 0, 255]
for i in range(len(contours)):
    r = np.random.randint(256)
    g = np.random.randint(256)
    b = np.random.randint(256)
    dst[markers == i + 1] = [b, g, r]
dst = cv2.addWeighted(src, 0.4, dst, 0.6, 0)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
