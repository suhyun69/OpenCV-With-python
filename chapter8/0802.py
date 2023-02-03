import cv2
import numpy as np

#1
# cv2.cornerEigenValsAndVecs()로 gray 영상에서 각 화소 이웃에 의한 2x2 공분산 행렬 M의 고유값과 고유 벡터를 res에 계산한다.
# res.shape=(512, 512, 6)dlek.
# cv2.split()로 res를 채널 분리하여 eigen에 저장한다.
# eigen[0] = ㅅ1, eigen[1] = ㅅ2의 고유값이고, ㅅ1에 대한 고유 벡터는 eigen[2] = x1, eigen[3] = y1이고, ㅅ2에 대한 고유 벡터는 eigen[4] = x2, eigen[5] = y2에 저장된다.
src = cv2.imread('../data/CornerTest.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
res = cv2.cornerEigenValsAndVecs(gray, blockSize=5, ksize=3)
print('res.shape=', res.shape)
eigen = cv2.split(res)

#2
# cv2.threshold()로 eigen[0]에서 임계값으로 T=0.2로 이진 영상 edge를 검출한다.
T = 0.2
ret, edge = cv2.threshold(eigen[0], T, 255, cv2.THRESH_BINARY)
edge = edge.astype(np.uint8)

#3
# 작은 고유값 eigien[1]이 T보다 크면, 큰 고유값 eigen[0]은 T보다 크므로 np.argwhere()로 eigen[1] > T인 좌표를 코너점 배열 corners에 검출하고, corners[:,[0,1]] = corners[:,[1,0]]에 의해 좌표순서를 열 x, 행 y로 변경하여 반환한다.
# corners의 각 코너점 좌표에 cv2.circle()로 dst에 반지름 5인 빨간색 원을 표시한다.
corners = np.argwhere(eigen[1] > T)
corners[:,[0,1]] = corners[:,[1,0]] # switch x, y
print('len(corners)=', len(corners))

dst = src.copy()
for x, y in corners:
    cv2.circle(dst, (x,y), 5, (0,0,255), 2)

cv2.imshow('edge', edge)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()