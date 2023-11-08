import cv2
import numpy as np

src = cv2.imread('../data/chessBoard.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

#1
# cv2.GFTTDetector 객체 goodF를 생성한다
# goodF.detect()로 gray에서 특징점 kp를 검출한다. len(kp) = 114개의 코너점을 검출한다
# cv2.drawKeypoints() 특징점 kp를 dst에 빨간색 원으로 같이 표시한다
goodF = cv2.GFTTDetector_create()
kp = goodF.detect(gray)
print('len(kp)=', len(kp))
dst = cv2.drawKeypoints(gray, kp, None, color=(0, 0, 255))
cv2.imshow('dst', dst)

#2
# maxCorners = 50, qualityLevel = 0.1, minDistance = 10, useHarrisDetector = True인 객체 cv2.GFTTDetector 객체 goodF2를 생성한다
# goodF2.detect()로 gray에서 특징점 kp를 검출한다. len(kp) = 38개의 검은색 사각형의 코너점을 검출한다
# cv2.drawKeypoints() 특징점 kp2를 dst2에 빨간색 원으로 같이 표시한다
goodF2 = cv2.GFTTDetector_create(maxCorners = 50, qualityLevel = 0.1, minDistance = 10, useHarrisDetector = True)
kp2 = goodF2.detect(gray)
print('len(kp2)=', len(kp2))
dst2 = cv2.drawKeypoints(gray, kp2, None, color=(0, 0, 255))
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()