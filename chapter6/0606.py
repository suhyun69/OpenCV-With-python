import cv2
import numpy as np

#1
# 입력 영상 src를 부드럽게 하여 미분 오차를 줄이기 위하여 ksize=(7,7)의 가우시안 블러 영상 blur를 생성하고, 라플라시안 필터를 적용하여 lap을 생성한다.
src = cv2.imread('../data/rect.jpg', cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(src, ksize=(7,7), sigmaX=0.0)
lap = cv2.Laplacian(blur, cv2.CV_32F, 3)

#2
# SGN(lap[y,x])와 8-이웃 화서 neighbors 중 최소값의 부호 SGN(mValue)가 같지 않으면 0-교차점으로 판단한다.
def SGN(x):
    if x >= 0:
        sign = 1
    else:
        sign = -1
    return sign

def zeroCrossing(lap):
    height, width = lap.shape
    Z = np.zeros(lap.shape, dtype=np.uint8)
    for y in range(1, height-1):
        for x in range(1, width-1):
            neighbors = [lap[y-1,x], lap[y+1,x], lap[y, x-1], lap[y,x+1], lap[y-1, x-1], lap[y-1, x+1], lap[y+1, x-1], lap[y+1, x+1]]
            mValue = min(neighbors)
            if SGN(lap[y,x]) != SGN(mValue):
                Z[x,y] = 255
    return Z

edgeZ = zeroCrossing(lap)
cv2.imshow('Zero Crossing', edgeZ)
cv2.waitKey()
cv2.destroyAllWindows()