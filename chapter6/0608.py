import cv2
import numpy as np

src = cv2.imread('../data/lena.jpg', cv2.IMREAD_GRAYSCALE)

#1
# logFilter() 함수는 ksize의 LoG(Laplacian of Gaussian) 필터를 생성하여 반환한다
# 라플라시안 필터링은 2차 미분을 사용하여 잡음 noise에 민감하다.
# 잡음을 줄이는 방법으로 입력 영상을 가우시안 필터링하여 잡음을 제거한 후에 라플라시안을 적용하는 방법을 사용할 수 있다
# 또는 가우시안 함수에 대한 2차 미분에 의한 라플라시안을 계산하여 윈도우 필터를 생성하여 필터링할 수도 있다.
# 이러한 필터링을 LoG(Laplacian of Gaussian)라 한다.
# 윈도우 필터의 크기는 n=2*3*x+1 또는 x = 0.3(n/2-1)+0.8로 계산할 수 있다.
# 에지는 LoG 필터링된 결과에서 0-교차하는 위치이다.
def logFilter(ksize = 7):
    k2 = ksize // 2
    sigma = 0.3 * (k2 - 1) + 0.8
    print('sigma=', sigma)
    LoG = np.zeros((ksize, ksize), dtype=np.float32)
    for y in range(-k2, k2+1):
        for x in range(-k2, k2+1):
            g = -(x*x + y*y) / (2.0 * sigma ** 2.0)
            LoG[y + k2, x + k2] = -(1.0 + g) * np.exp(g) / (np.pi * sigma ** 4.0)
    return LoG

#2
# logFilter()로 ksize의 가우시안의 라플라시안 필터 kernel을 생성한다.
# cv2.filter2D()로 src에 kernel 필터를 적용하여 LoG를 생성한다.
kernel = logFilter()
LoG = cv2.filter2D(src, cv2.CV_32F, kernel)
cv2.imshow('LoG', LoG)

#3
# zeroCrossing2() 함수는 lap[y,x]의 8-이웃 화소 neighbors에서 임계값 thresh를 이용하여
# value > thresh인 개수는 pos에, value < -thresh는 neg에 카운트하여
# pos > 0 and neg > 0이면 0-교차점으로 판단한다.
# 0 대신 thresh를 사용하는 것은 계산에서 실수에 따른 오차 때문이다
def zeroCrossing2(lap, thresh = 0.01):
    height, width = lap.shape
    Z = np.zeros(lap.shape, dtype=np.uint8)
    for y in range(1, height-1):
        for x in range(1, width-1):
            neighbors = [lap[y-1,x], lap[y+1,x], lap[y, x-1], lap[y,x+1], lap[y-1, x-1], lap[y-1, x+1], lap[y+1, x-1], lap[y+1, x+1]]
            pos = 0
            neg = 0
            for value in neighbors:
                if value > thresh:
                    pos += 1
                if value < -thresh:
                    neg += 1
            if pos > 0 and neg > 0:
                Z[y,x] = 255
    return Z

edgeZ = zeroCrossing2(LoG)
cv2.imshow('Zero Crossing2', edgeZ)
cv2.waitKey()
cv2.destroyAllWindows()
