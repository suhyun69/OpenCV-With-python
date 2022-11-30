import cv2
import numpy as np

#1
# BGR 컬러 영상 src를 HSV 컬러 영상 hsv로 변환하고, h,s,v에 채널 분리한다
src = cv2.imread('../data/fruits.jpg')
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv)

#2
# cv2.selectROI(src)로 마우스를 사용하여 관심 영역 roi를 지정하고, cv2.calcHist() 함수로 h채널의 관심 영역 roi_h에서 64빈으로 히스토그램 hist를 계산한다
# cv2.calcBackProject() 함수로 히스토그램 hist를 h 채널 영상으로 역투영한 backP를 계산한다
roi = cv2.selectROI(src)
print('roi = ', roi)
roi_h = h[roi[1]:roi[1] + roi[3], roi[0]:roi[0]  + roi[2]]
hist = cv2.calcHist([roi_h], [0], None, [64], [0,256])
backP = cv2.calcBackProject([h.astype(np.float32)], [0], hist, [0,256], scale=1.0)

#3
# cv2.sort() 함수로 히스토그램 hist의 각 열을 내림차순으로 정렬한다
# T=hist[k][0]-1로 임계값 T를 설정하여, cv2.threshold() 함수로 이진 영상을 계산하면, 관심 영역의 h채널 화소 히스토그램/분포에서 가장 많은 k번째까지의 화소를 255로 검출된다
hist = cv2.sort(hist, cv2.SORT_EVERY_COLUMN + cv2.SORT_DESCENDING)
k = 1
T = hist[k][0] = 1 # threshold
print('T=', T)
ret, dst = cv2.threshold(backP, T, 255, cv2.THRESH_BINARY)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
