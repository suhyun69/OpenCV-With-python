import cv2
import numpy as np

src = cv2.imread('../data/chessBoard.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

#1
# 안정적인 그레이 레벨의 단계의 간격인 _delta=10으로 cv2.MSER 객체 mserF를 생성한다
# mserF.detect()로 gray에서 영역의 중심점인 특징점 kp를 검출하고, cv2.drawKeypoints()로 특징점 kp를 dst에 빨간색 원으로 표시한다
# 특징점이 유사 지점에서 중복으로 검출되어 len(kp) = 202개이다
# filteringByDistance() 함수를 사용하면 가까운 거리의 중복 검출되는 특징점을 제거할 수 있다
mserF = cv2.MSER_create(10)
kp = mserF.detect(gray)
print('len(kp)=', len(kp))
dst = cv2.drawKeypoints(gray, kp, None, color=(0,0,255))
cv2.imshow('dst', dst)

#2
# mserFdetectRegions()로 gray에서 특징영역의 좌표를 regions에 검출한다.
# len(regions) = 202개이다.
# cv2.convexHull()로 regions[i]의 영역을 볼록다각형 hulls[i]에 계산한다
# cv2.polylines()로 dst2에 hulls를 초록색 다각형으로 표시한다
dst2 = dst.copy()
regions, bboxes = mserF.detectRegions(gray)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
cv2.polylines(dst2, hulls, True, (0, 255, 0))
cv2.imshow('dst2', dst2)

#3
# cv2.fitEllipse()로 regions의 각 영역 좌표 pts를 타원 근사한 box를 dst3에 파란색 타원으로 표시하고
# bboxes[i]를 초록색 사각형으로 dst3에 표시한다
dst3 = dst.copy()
for i, pts in enumerate(regions):
    box = cv2.fitEllipse(pts)
    cv2.ellipse(dst3, box, (255, 0, 0), 1)
    x, y, w, h = bboxes[i]
    cv2.rectangle(dst3, (x, y), (x + w, y + h), (0, 255, 0))
cv2.imshow('dst3', dst3)
cv2.waitKey()
cv2.destroyAllWindows()