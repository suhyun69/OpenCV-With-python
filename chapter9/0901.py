import cv2
import numpy as np

src = cv2.imread('../data/chessBoard.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

#1
# 임계값 threshold = 30인 cv2.FastFeatureDetector 클래스 객체 fastF를 생성한다.
# fastF.detect()로 gray에서 특징점 kp를 검출하고, cv2.drawKeypoints()로 특징점 kp를 dst에 빨간색 원으로 표시한다
# 특징점의 개수 len(kp) = 98개이다.
fastF = cv2.FastFeatureDetector.create(threshold = 30)
kp = fastF.detect(gray)
dst = cv2.drawKeypoints(gray, kp, None, color=(0,0,255))
print('len(kp)=', len(kp))
cv2.imshow('dst', dst)

#2
# fastF.setNonmaxSuppression(False)로 지역 극값 억제를 하지 않고 fastF.detect()로 gray에서 특징점 kp2를 검출하면 특징점의 개수는 len(kp2) = 867개로 특징점의 개수가 증가한다
# cv2.drawKeypoints()로 특징점 kp2를 dst에 빨간색 원으로 표시한다
fastF.setNonmaxSuppression(False)
kp2 = fastF.detect(gray)
dst2 = cv2.drawKeypoints(src, kp2, None, color=(0,0,255))
print('len(kp2)=', len(kp2))
cv2.imshow('dst', dst2)

#3
# cv2.KeyPoint_convert()로 특징점 kp를 좌표 리스트 points로 변환하여, cv2.circle()로 src를 복사한 dst3에 파란색 원으로 표시한다
dst3 = src.copy()
points = cv2.KeyPoint_convert(kp)
points = np.int32(points)

for cx, cy in points:
    cv2.circle(dst3, (cx, cy), 3, color=(255,0,0), thickness=1)
cv2.imshow('dst3', dst3)
cv2.waitKey()
cv2.destroyAllWindows()