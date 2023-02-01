import cv2
import numpy as np

#1
# 컬러 입력 영상 src를 그레이스케일 영상 gray로 변환한다.
# 입력 영상에서 원이 검은색, 배경의 흰색이어서 cv2.THRESH_BINARY_INV로 임계값 128을 적용하여 이진 영상을 생성한다.
src = cv2.imread('../data/circles.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, res = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

#2
# cv2.connectedComponents()로 이진 영상 res를 레이블링하여 레이블 개수는 배경을 포함하여 ret = 4이고, 레이블 정보 labels를 생성한다.
# 검출된 물체인 원의 개수는 ret - 1이다.
ret, labels = cv2.connectedComponents(res)
print('ret=', ret)

#3
# labels에서 배경 레이블(0)은 제외하고, 1에서부터 ret-1까지의 레이블 영역을 난수로 생성한 같은 컬러로 지정하여 dst 영상을 생성한다.
dst = np.zeros(src.shape, dtype=src.dtype)
for i in range(1, ret):
    r = np.random.randint(256)
    g = np.random.randint(256)
    b = np.random.randint(256)
    dst[labels == i] = [b, g, r]

cv2.imshow('res', res)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()