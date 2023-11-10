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
# cv2.KAZE_create()로 KAZE 객체 kazeF를 생성한다.
# cv2.AKAZE_create()로 KAZE 객체 akazeF를 생성한다.
# kazeF.detect()로 gray에서 len(kp) = 24개의 특징점을 kp에 검출한다.
# akazeF.detect()를 사용하면 len(kp) = 32개의 특징점을 kp에 검출한다.
# cv2.drawKeypoints()로 특징점 kp를 dst에 빨간색 원으로 표시한다
src = cv2.imread('../data/cornerTest.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0.0)

kazeF = cv2.KAZE_create()
akazeF = cv2.AKAZE_create()
kp = kazeF.detect(gray)
# kp = akazeF.detect(gray)
print('len(kp)=', len(kp))
dst = cv2.drawKeypoints(gray, kp, None, color=(0,0,255))
cv2.imshow('dst', dst)

#3
# sorted()로 kp를 반응값 기준으로 내림차순 정렬하고,  distE = 10 이내의 특징점을 제거하여 len(filtered_kp)=8개를 filtered_kp에 검출한다
# kazeF.compute()로 gray에서 특징점 filtered_kp를 이용하여 KAZE 디스크립터 des를 계산한다
# des.dtype = floaty32이고, des.shape=(8,64)이다.
# akazeF.compute()를 사용하여 디스크립터 des를 계산하면 des.dtype = 'uint8'이고, des.shape=(8,64)이다.
kp = sorted(kp, key=lambda f: f.response, reverse=True)
# filtered_kp = list(filter(lambda f: f.response > 0.01, kp))
filtered_kp = filteringByDistance(kp, 10)
print('len(filtered_kp)=', filtered_kp)

kp, des = kazeF.compute(gray, filtered_kp)
print('des.shape=', des.shape)
print('des.dtype=', des.dtype)
print('des=', des)

#4
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
