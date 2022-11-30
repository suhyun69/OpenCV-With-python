import cv2
import numpy as np

# BGR 컬러 영상 src를 BGR컬러 영상의 히스토그램 평활화는 HSV, YCrCb 등의 컬러 모델로 변환한 다음, 밝기값 채널(V,Y)에 히스토그램 평활화를 적용하고 BGR 영상으로 변환한다

src = cv2.imread('../data/lena.jpg')
cv2.imshow('src', src)

#1
# cv2.cvtColor()로 BGR 컬러 영상 dst를 HSV 컬러 영상 hsv로 변환하고 cv2.split()로 hsv를 h,s,v에 채널 분리한다
# cv2.equalizeHist()로 v를 v2에 평활화한다
# cv2.merge()로 [h,s,v]를 hsv2에 채널 합성한다
# cv2.cvtColor()로 HSV 컬러 영상 hsv2를 dst에 변환한다
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv)

v2 = cv2.equalizeHist(v)
hsv2 = cv2.merge([h,s,v2])
dst = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
cv2.imshow('dst', dst)

#2
yCrCv = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
y,Cr,Cv = cv2.split(yCrCv)

y2 = cv2.equalizeHist(y)
yCrCv2 = cv2.merge([y2, Cr, Cv])
dst2 = cv2.cvtColor(yCrCv2, cv2.COLOR_YCrCb2BGR)

cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()
