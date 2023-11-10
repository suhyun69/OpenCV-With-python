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
# 입력 영상 src의 그레이스케일 영상 gray에 가우시안 블러링을 수행한다. 가우시안 블러링 수행 여부에 따라 결과가 약간 다를 수 있다
# cv2.ORB_create(scoreType = 1)로 FAST_SCORE를 사용하여 orbF 객체를 생성한다.
# nfeatures로 orbF 객체를 생성한다.
# orbF.detect()로 grayt에서 len(kp) = 63개의 특징점을 kp에 검출한다.
# cv2.drawKeypoints()로 특징점 kp를 dst에 빨간색 원으로 표시한다.
src = cv2.imread('../data/cornerTest.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0.0)

## orbF = cv2.ORB_create() # HARRIS_SCORE
orbF = cv2.ORB_create(scoreType = 1) # FAST_SCORE
kp = orbF.detect(gray)
print('len(kp)=', len(kp))
dst = cv2.drawKeypoints(gray, kp, None, color=(0,0,255))
cv2.imshow('dst', dst)

#3
# sorted()로 kp를 반응값 기준으로 내림차순 정렬하고, 반응값이 50보다 작은 특징점을 제거하고 distE = 10 이내의 특징점을 제거하여 len(filtered_kp) = 8인 filtered_kp를 생성한다.
# orbF.compute()로 gray에서 특징점 filtered_kp를 이용하여 ORB 디스크럽터 des를 계산한다
# des.dtype = uint8이고, des.shape = (8, 32)이다. 즉, 디스크립터는 8개의 특징점 가각에 대하여 32바이트이다.
kp = sorted(kp, key=lambda f: f.response, reverse=True)
filtered_kp = list(filter(lambda f: f.response > 50, kp))
filtered_kp = filteringByDistance(kp, 10)
print('len(filtered_kp)=', len(filtered_kp))

kp, des = orbF.compute(gray, filtered_kp)
print('des.shape=', des.shape)
print('des=', des)

#4
# cv2.drawKeypoints()로 특징점 filtered_kp를 dst2에 빨간색 원으로 표시한다.
# for 문으로 filtered_kp의 각 특징점 f에 대해, 좌표는 x,y, 크기는 size에 읽고, rect = ((x,y), (size, size), f.angel)로 회전 사각형을 정의하고
# cv2.boxPoints()로 rect의 모서리 좌표를 box에 읽고,
# cv2.polylines()로 dst2에 초록색으로 그린다.
# cv2.circle()로 dst2에 파란색 원으로 표시한다.
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


