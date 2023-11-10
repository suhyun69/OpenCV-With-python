import cv2
import numpy as np

#1
# filteringByDistance()는 9.2에서 작성한 특징점 kp에서 거리 오차 distE 보다 작은 특징점을 제거하는 함수이다.
def distance(f1, f2):
    x1, y1 = f1.pt
    x2, y2 = f2.pt
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def filteringByDistance(kp, distE=0.5):
    size = len(kp)
    mask = np.arange(1, size+1).astype(np.bool8)
    for i, f1 in enumerate(kp):
        if not mask[i]:
            continue
        else:
            for j, f2 in enumerate(kp):
                if i == j:
                    continue
                if distance(f1, f2) < distE:
                    mask[j] = False
    np_kp = np.array(kp)
    return list(np_kp[mask])

#2
# 입력 영상 src의 그레이스케일 영상 gray에 가우시안 블러링을 수행한다.
# FastFeatureDetector 객체 fastF, MSER 객체 mserF, SimpleBlobDetector 객체 blobF, GFTTDetector 객체 goodF를 생성한다
# 특징점 객체 fastF, mserF, blobF, goodF 중 하나로 특징점 kp를 생성한다.
# filteringByDistance()로 kp를 거리가 10보다 작은 특징점을 제거하여 filtered_kp를 생성한다.
# 반응값이 0인 특징값이 있기 때문에 여기서는 반응값으로 정렬하지 않았다.
# 따라서 특징값의 순서에 따라 먼저 나오는 특징값을 기준으로 거리가 가까운 주위의 특징을 삭제한다.
# cv2.drawKeypoints()로 특징점 kp를 dst에 빨간색 원으로 표시한다.
src = cv2.imread('../data/chessBoard.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0.0)

fastF = cv2.FastFeatureDetector_create(threshold = 30)
mserF = cv2.MSER_create(10)
blobF = cv2.SimpleBlobDetector_create()
goodF = cv2.GFTTDetector_create(maxCorners = 20, minDistance = 10)

kp = fastF.detect(gray)
# kp = mserF.detect(gray)
# kp = blobF.detect(gray)
# kp = goodF.detect(gray)
print('len(kp)=', len(kp))

filtered_kp = filteringByDistance(kp, 10)
print('len(filtered_kp)=', len(filtered_kp))
dst = cv2.drawKeypoints(gray, filtered_kp, None, color=(0,0,255))
cv2.imshow('dst', dst)

#3
# orbF.compute()로 gray에서 특징점 filtered_kp를 이용하여 ORB 디스크립터 des를 계산한다
# des.shape = (len(filtered_kp), 32)이다. 즉, 디스크립터는 len(filtered_kp)개의 특징점 각가에 대하여 32바이트이다.
# cv2.drawKeypoints()로 특징점 filtered_kp를 dst2에 빨간색 원으로 표시한다
# for 문으로 filtered_kp의 각 특징점 f에 대해, 특징의 크기와 각도를 이용하여 회전 사각형과 원을 dst2에 초록색으로 표시한다
orbF = cv2.ORB_create()
filtered_kp, des = orbF.compute(gray, filtered_kp)
print('des.shape=', des.shape)
print('des=', des)

dst2 = cv2.drawKeypoints(gray, filtered_kp, None, color=(0,0,255))
for f in filtered_kp:
    x, y = f.pt
    size = f.size
    rect = ((x,y), (size, size), f.angle)
    box =cv2.boxPoints(rect).astype(np.int32)
    cv2.polylines(dst2, [box], True, (0, 255, 0), 2)
    cv2.circle(dst2, (round(x), round(y)), round(f.size/2), (255,0,0), 2)

cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()
