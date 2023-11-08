import cv2
import numpy as np

src = cv2.imread('../data/chessBoard.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

#1
# cv2.FastFeatureDetector 객체 fastF를 생성한다. 기본 임계값은 threshold = 1이다.
# fastF.detect()로 gray에서 특징점 kp를 검출하고, cv2.drawKeypoints()로 특징점 kp를 dst에 파란색 원으로 표시한다.
# 특징저의 개수는 len(kp) = 167개이다. 많은 특징점이 유사한 위치에서 검출되는 것을 확인할 수 있다.
# 특징점의 반응값, 거리 등을 이용하여 삭제할 수 있다
fastF = cv2.FastFeatureDetector_create()
kp = fastF.detect(gray)
dst = cv2.drawKeypoints(gray, kp, None, color=(255, 0, 0))
print('len(kp)=', len(kp))

#2
# sorted()로 특징점 kp를 반응값 기준으로 내림차순으로 정렬한다
# cv2.drawKeypoints()로 반응값 기준으로 내림차순으로 정렬한 특징점에서 반응값이 큰 10개 kp[:10]을 모든 특징점을 파란색 원으로 표시한 dst에 빨간색 원을 추가하여 표시한다
kp = sorted(kp, key = lambda f: f.response, reverse = True)
cv2.drawKeypoints(gray, kp[:10], dst, color=(0,0,255), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
cv2.imshow('dst', dst)

#3
# OpenCV_Python은 KeyPointsFilter를 사용할 수 없기 때문에 직접 특징점 필터링을 작성하였다.
# filter()로 반응값이 50보다 작은 특징은 제거하여 kp2를 생성한다.
# 특징점의 개수는 len(kp2) = 91개이다. cv2.drawKeypoints()로 특징점 kp2를 dst2에 빨간색 원으로 표시한다
kp2 = list(filter(lambda f: f.response > 50, kp))
print('len(kp2)=', len(kp2))

dst2 = cv2.drawKeypoints(gray, kp2, None, color=(0,0,255))
cv2.imshow('dst2', dst2)

#4
# filterByDistance 함수는 반응값 기준으로 내림차순 정렬된 특징점 kp에서 거리 오차 distE보다 작은 특징점은 삭제한다
# kp3 = filteringByDistance()로 kp2에서 거리오차 distE = 30보다 작은 특징점을 삭제하여 특징점의 개수는 len(kp2) = 38개이다.
# drawKeypoints()로 특징점 kp3을 dst3에 빨간색 원으로 표시한다.
def distance(f1, f2):
    x1, y1 = f1.pt
    x2, y2 = f2.pt
    return np.sqrt((x2-x1) ** 2 + (y2-y1) ** 2)

def filterByDistance(kp, distE = 0.5):
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

kp3 = filterByDistance(kp2, 30)
print('len(kp3)=', len(kp3))
dst3 = cv2.drawKeypoints(gray, kp3, None, color=(0,0,255))
cv2.imshow('dst3', dst3)
cv2.waitKey()
cv2.destroyAllWindows()