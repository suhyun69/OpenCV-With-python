import cv2
import numpy as np

#1
# findLocalMaxima()는 src에서 팽창과 침식의 모폴로지 연산으로 지역 극대값의 좌표를 points 배열에 검출하여 반환한다.
# cv2.dilate()로 src에서 rectKernel의 이웃에서 최대값을 dilate에 계산한다.
# 커널을 None으로 사용하면 3x3 사각형 이웃이다.
# src == dilate로 src에서 지역 최대값의 위치를 localMax 배열에 계산한다.
# cv2.erode()로 src에서 rectKernel의 이웃에서 최소값을 erode에 계산한다.
# localMax2 = src > erode로 최소값보다 큰 위치를 localMax2에 계산한다.
# localMax &= localMax2로 localMax와 localMax2를 논리곱하여 지역 최대값 위치를 localMax 배열에 계산한다.
# points = np.argwhere()로 localMax 배열에서 True인 위치의 좌표를 points 배열에 찾는다.
# np.argwhere()는 행, 열 순서로 찾기 때문에, points[:,[0:1]] = points[:,[1,0]]에 의해 좌표순서를 열(x), 행(y)로 변경하여 반환한다.
def findLocalMaxima(src):
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(11,11))

    # local max if kensel = None, 3x3
    dilate = cv2.dilate(src, kernel)
    localMax = (src == dilate)

    # local min if kernel = Noen, 3x3
    erode = cv2.erode(src, kernel)
    localMax2 = src > erode
    localMax &= localMax2
    points = np.argwhere(localMax == True)
    points[:,[0,1]] = points[:,[1,0]] # switch x, y
    return points

#2
# 그레이스케일 영상 gray에서 cv2.preCornerDetect()로 res를 계산한다. 극대값만을 찾기 위하여 np.abs(res)인 절대값 배열에서, cv2.threshold()로 임계값 thresh = 0.1보다 작은 값은 0으로 변경하여 res2에 저장한다.
# 즉, res에서 임계값보다 작은 값을 제거한다
# findLocalMaxima()로 res2에서 지역 극값의 좌표를 코너점으로 찾아 corners에 저장한다
src = cv2.imread('../data/CornerTest.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
res = cv2.preCornerDetect(gray, ksize=3)
ret, res2 = cv2.threshold(np.abs(res), 0.1, 0, cv2.THRESH_TOZERO)
corners = findLocalMaxima(res2)
print('corners.shape=', corners.shape)

#3
# src를 dst에 복사하고, 코너점 배열 corners의 각 코너점 좌표에 cv2.circle()로 dst에 반지름 5, 빨간색 원을 그린다.
dst = src.copy()
for x, y in corners:
    cv2.circle(dst, (x,y), 5, (0, 0, 255), 2)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()