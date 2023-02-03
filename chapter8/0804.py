import cv2
import numpy as np

#1
# 예제 8.1에서 설명한 findLocalMaxima()는 src에서 팽창과 침식의 모폴로지 연산으로 지역 극대값의 좌표를 points를 검출하여 반환한다.
def findLocalMaxima(src):
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(11,11))

    dilate = cv2.dilate(src, kernel) # local max if kensel = None, 3x3
    localMax = (src == dilate)

    erode = cv2.erode(src, kernel) # local min if kernel = Noen, 3x3
    localMax2 = src > erode
    localMax &= localMax2
    points = np.argwhere(localMax == True)
    points[:,[0,1]] = points[:,[1,0]] # switch x, y
    return points

#2
# cv2.cornerHarris()로 gray 영상에서 각 화소 이웃에 의한 2x2 공분산 행렬 M의 Harris 반응값을 res에 계산한다.
# res.shape=(512, 512)이다.
# cv2.threshold()로 np.abs(res)에서 임계값 thresh=0.02보다 작은 값을 0으로 변경하여 res에 저장한다.
# Harris 반응값 res를 [0, 255] 범위로 정규화한다.
src = cv2.imread('../data/CornerTest.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
res = cv2.cornerHarris(gray, blockSize=5, ksize=3, k=0.01)
ret, res = cv2.threshold(np.abs(res), 0.02, 0, cv2.THRESH_TOZERO)
res8 = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imshow('res8', res8)
corners = findLocalMaxima(res)
print('corners.shape=', corners.shape)

#3
# findLocalMaxima()로 res에서 코너점을 찾아 corners에 저장한다.
# 이때의 코너점 corners는 정수 좌표이다.
# corners를 np.float32 자료형으로 변환한다.
# 이때 order='C'에 의해 C언어 스타일 메모리 구조를 지정하거나, 복사하지 않으면 cv2.cornerSubPix()에서 오류가 발생함에 주의한다.
# cv2.cornerSubPix()로 gray 영상에서 코너점 좌표 corners를 부화소 수준으로 계산하여 corners에 저장한다.
# 실행 결과를 보면 corners2의 좌표는 corners에서 약간 이동된 다른 것을 알 수 있다.
# corners2의 각 코너점 좌표에 cv2.circle()로 반지름 5인 빨간색 원을 표시한다.
corners = corners.astype(np.float32, order='C')
term_crit = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 0.01)
corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term_crit)
print('corners2=', corners2)

dst = src.copy()
for x, y in np.int32(corners2):
    cv2.circle(dst, (x,y), 3, (0,0,255), 2)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()