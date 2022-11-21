import cv2
import numpy as np

src = cv2.imread('../data/lena.jpg')
b, g, r = cv2.split(src)

# 원본 컬러 영상 src의 채널 분리 영상
cv2.imshow('b', b)
cv2.imshow('g', g)
cv2.imshow('r', r)

X = src.reshape(-1, 3) # 모양을 재조정하여 X의 각 행에 화소의 컬러값을 위치시킨다
print('X.shpae = ', X.shape)

mean, eVects = cv2.PCACompute(X, mean = None) # X의 평균 벡터 mean, 공분산 행렬의 고유 벡터 eVects를 계산한다
print('mean = ', mean)
print('eVects = ', eVects)

Y = cv2.PCAProject(X, mean, eVects) # 고유 벡터 eVects에 의해 X를 Y에 PCA 투영한다.
Y = Y.reshape(src.shape)
print('Y.shape = ', Y.shape)

eImage = list(cv2.split(Y))
for i in range(3):
    cv2.normalize(eImage[i], eImage[i], 0, 255, cv2.NORM_MINMAX)
    eImage[i] = eImage[i].astype(np.uint8)

# 컬러의 공분산 행렬의 고유 벡터에 의한 PCA 투영 영상
cv2.imshow('eImage[0]', eImage[0]) # 고유값이 가장 큰 고유 벡터(가장 큰 축)로의 투영으로 정보가 가장 많다
cv2.imshow('eImage[1]', eImage[1])
cv2.imshow('eImage[2]', eImage[2]) # 고유값이 가장 작은 고유 벡터(가장 작은 축)로의 투영으로 정보가 가장 적다
cv2.waitKey()
cv2.destroyAllWindows()