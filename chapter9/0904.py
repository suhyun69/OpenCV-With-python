import cv2
import numpy as np

src = cv2.imread('../data/chessBoard.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

#1
# cv2.SimpleBlobDetector_Params()로 params 객체를 생성하고 속성을 설정한다
params = cv2.SimpleBlobDetector_Params()
params.blobColor = 0
params.thresholdStep = 5
params.minThreshold = 20
params.maxThreshold = 100
params.minDistBetweenBlobs = 5
params.filterByArea = True
params.minArea = 25
params.maxArea = 5000
params.filterByConvexity = True
params.minConvexity = 0.89

#2
# cv2.SimpleBlobDetector 객체 blobF를 생성한다. blobF.detect()로 gray에서 특징점 kp를 검출한다
# blobF.detect(src)로 컬러 영상 src에서도 특징점을 검출할 수 있다.
# len(kp) = 14개의 검은색 영역을 모두 검출한다
# cv2.drawKeypoints()로 특징점 kp를 dst에 빨간색 원으로 표시한다
# blobF = cv2.SimpleBlobDetector_create(params)
blobF = cv2.SimpleBlobDetector_create()
kp = blobF.detect(gray)
print('len(kp)=', len(kp))
dst = cv2.drawKeypoints(gray, kp, None, color=(0, 0, 255))

#3
# 특징점 kp의 각 특징점 f에서 반지름을 r = round(f.size/2)로 계산하고, cx, cy = f.pt로 중심점을 계산하여 cv2.circle()로 dst에 파란색 원으로 표시한다
for f in kp:
    r = int(f.size/2)
    cx, cy = f.pt
    cv2.circle(dst, (round(cx), round(cy)), r, (0, 0, 255), 2)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
