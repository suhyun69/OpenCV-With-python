import cv2
import numpy as np
from matplotlib import pyplot as plt

src = cv2.imread('../data/rect.jpg', cv2.IMREAD_GRAYSCALE)
# src = cv2.imread('../data/line.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('src', src)

#1
# cv2.Sobel() 함수를 이용하여 입력 영상 src의 그래디언트 gx, gy를 계산한다.
# cv2.cartToPolar()로 그래디언트 크기 mag와 각도 angle을 계산한다
gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)

mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(angle)
print('mag:', minVal, maxVal, minLoc, maxLoc)

#2
# cv2.threshold()로  mag에서 임계값 100을 사용하여 이진 영상 edge를 계산한다.
# 화면표시를 위해 화소 자료형을 np.uint8로 변경한다
ret, edge = cv2.threshold(mag, 100, 255, cv2.THRESH_BINARY)
edge = edge.astype(np.uint8)
cv2.imshow('edge', edge)

#3
# angleM을 배경이 흰색(255, 255, 255)인 3-채널 컬러 영상으로 생성한다.
# edge[y,x] !=0에 의해 에지인화소에서 그래디언트 각도가 0, 90, 180, 270도인 화소에서 빨강, 초록, 파랑, 노랑, 그 외 각도의 에지 화소는 회색(gray)을 angleM[y,x]에 저장한다.
height, width = mag.shape[:2]
angleM = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)
for y in range(height):
    for x in range(width):
        if edge[y, x] != 0:
            if angle[y, x] == 0:
                angleM[y, x] = (0, 0, 255) # red
            elif angle[y, x] == 90:
                angleM[y, x] = (0, 255, 0) # green
            elif angle[y, x] == 180:
                angleM[y, x] = (255, 0, 0) # blue
            elif angle[y, x] == 270:
                angleM[y, x] = (0, 255, 255) # yellow
            else:
                angleM[y, x] = (128, 128, 128) # gray

cv2.imshow('angleM', angleM)
# cv2.waitKey()
# cv2.destroyAllWindows()

#4
# cv2.clacHist()로 그래디언트 각도 angle의 히스토그램 histSize=[360], ranges=[0, 360], mask=edge로 hist에 계산한다.
# 그래디언트 각도는 0, 90, 180, 270도에서 대부분이다
# 여기서, 변화가 없는 gx=0, gy=0인 화소와 구분하기 위하여, 에지 영상 edge를 마스크로 사용하여, 에지에서만 히스토그램을 구하는 것이 중요하다
# plt.bar()로 막대 그래프를 그리기 위하여 hist=hist.flatten()로 hist의 모양을 (360, )의 1차원 행 배열로 변경한다
hist = cv2.calcHist(images=[angle], channels=[0], mask=edge, histSize=[360], ranges=[0, 360])

hist = hist.flatten()
# plt.title('hist: binX = np.arange(360)')
plt.plot(hist, color='r')
binX = np.arange(360)
plt.bar(binX, hist, width=1, color='b')
plt.show()
