import cv2
import numpy as np

#1
# cv2.cornerHarris()로 gray 영상에서 각 화소 이웃에 의한 2x2 공분산 행렬 M의 Harris 반응값을 res에 게산한다
# res.shape=(512,512)이다
src = cv2.imread('../data/CornerTest.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
res = cv2.cornerHarris(gray, blockSize=5, ksize=3, k=0.01)

#2
# cv2.dilate()로 res에 3x3 사각형 커널을 사용하여 팽창연산으로 지역 최대값을 res에 계산한다.
# cv2.threshold()에 res에서 임계값 thresh = 0.01 * res.max()보다 크면 255인 이진 영상을 res에 저장한다.
# Harris 반응값 res를 np.uint8 자료형으로 변경한다.
res = cv2.dilate(res, None) # 3x3 rect kernel
ret, res = cv2.threshold(res, 0.01 * res.max(), 255, cv2.THRESH_BINARY)
res8 = np.uint8(res)
cv2.imshow('res8', res8)

#3
# cv2.connectedComponentsWithStats()로 이진 영상 res8를 레이블링하여 레이블 개수 ret, 레이블 정보 labels, 통계정보 stats, 중심점 centroids를 계산한다.
# 배경을 포함하기 때문에 ret = 9이다.
# centroids의 자료형을 np.float32로 변경한다
# 반응값이 임계값보다 큰 영역의 중심인 centroids가 코너점이다.
# 이때는 order='C'를 지정하지 않아도 cv2.cornerSubPix()에서 오류가 발생하지 않는다.
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(res8)
print('centroids.shape=', centroids.shape)
print('centroids=', centroids)
centroids = np.float32(centroids)

#4
# cv2.cornerSubPix()로 gray 영상에서 centroids를 부화소 수준으로 계산하여 corners에 저장한다.
# corners[0]은 배경의 중심점이다.
# 물체의 코너점인 corners[1:]의 각 좌표에 cv2.circle()로 반지름 5인 빨간색 원을 표시한다.
term_crit = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 0.01)
corners = cv2.cornerSubPix(gray, centroids, (5,5), (-1,-1), term_crit)
print('corners2=', corners)

#5
# cv2.cornerHarris()와 cv2.cornerSubPix()를 같이 사용하였지만, Harris 반응값에서 초기 코너점을 찾는 방법이 다르기 때문에 검출된 코너점이 약간 차이가 난다.
corners = np.round(corners)
dst = src.copy()
for x, y in corners[1:]:
    cv2.circle(dst, (int(x), int(y)), 5, (0,0,255), 2)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()