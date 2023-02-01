import cv2
import numpy as np

#1
# 컬러 입력 영상 src를 그레이스케일 영상 gray로 변환한다.
# 입력 영상에서 원이 검은색, 배경의 흰색이어서 cv2.THRESH_BINARY_INV로 임계값 128을 적용하여 이진 영상 res를 생성한다.
src = cv2.imread('../data/circles.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, res = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

#2
# cv2.connectedComponentsWithStats()로 이진 영상 res를 레이블링하여 레이블 개수 ret, 레이블 정보 labels, 통계정보 stats, 중심점 centroids를 계산한다
# ret = 4이다
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(res)
print('ret=', ret)
print('stats=', stats)
print('centroids=', centroids)

#3
# labels에서 배경 레이블(0)은 제외하고, 1에서부터 ret-1까지의 레이블 영역을 난수로 생성한 같은 컬러로 채운다
dst = np.zeros(src.shape, dtype=src.dtype)
for i in range(1, int(ret)):
    r = np.random.randint(256)
    g = np.random.randint(256)
    b = np.random.randint(256)
    dst[labels == i] = [b, g, r]

#4
# 레이블 i의 통계정보 stats[i]를 이용하여 바운딩 빨간색으로 사각형을 그리고, 중심점 centroids[i]를 이용하여 파란색으로 원을 그린다.
for i in range(1, int(ret)):
    x, y, width, height, area = stats[i]
    cv2.rectangle(dst, (x, y), (x + width, y + height), (0, 0, 255), 2)

    cx, cy = centroids[i]
    cv2.circle(dst, (int(cx), int(cy)), 5, (255, 0, 0), -1)

cv2.imshow('res', res)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()