import cv2
import numpy as np

#1
# cv2.cornerMinEigenVal()로 gray 영상에서 각 화소 이웃에 의한 2x2 공분산 행렬 M의 작은 고유값 s2를 eigen.shape = (512, 512)인 eigen에 계산한다.
src = cv2.imread('../data/CornerTest.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
eigen = cv2.cornerMinEigenVal(gray, blockSize=5)
print('eigen.shape=', eigen.shape)

#2
# np.argwhere()로 eigen > T인 좌표를 코너점 배열 corners에 검출하고, corners[:,[0,1]] = corners[:,[1,0]]에 의해 좌표순서를 열 x, 행 y로 변경하여 반환한다.
# corners의 각 코너점 좌표에 cv2.circle()로 dst에 반지름 5인 빨간색 원을 표시한다.
T = 0.2
corners = np.argwhere(eigen > T)
corners[:,[0,1]] = corners[:,[1,0]] # switch x, y
print('len(corners)=', len(corners))

dst = src.copy()
for x, y in corners:
    cv2.circle(dst, (x,y), 3, (0,0,255), 2)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()