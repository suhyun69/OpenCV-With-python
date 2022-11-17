import cv2
import numpy as np

src = cv2.imread('../data/lena.jpg', cv2.IMREAD_GRAYSCALE)
dst = np.zeros(src.shape, dtype=src.dtype)

# 원본 영상 src와 같은 자료형과 크기이고, 0으로 초기화된 dst를 생성하고,
# for문에서 roi를 이용하여, 하나의 크기가 w x h인 N x N 블록 평균 영상을 dst에 계산한다

N = 4 # 8, 32, 64
height, width = src.shape # 그레이스케일 영상
# height, width, channel = src.shape # 컬러 영상

h = height // N
w = width // N
for i in range(N):
    for j in range(N):
        y = i * h
        x = j * w
        roi = src[y:y+h, x:x+w]
        dst[y:y+h, x:x+w] = cv2.mean(roi)[0] # 그레이스케일 영상
        # dst[y:y+h, x:x+w] = cv2.mean(roi)[0:3] # 컬러 영상

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()