import cv2
import numpy as np

X = np.array([[0, 0, 0, 100, 100, 150, -100, -150],
              [0, 50, -50, 0, 30, 100, -20, -100]],
             dtype=np.float64)
X = X.transpose() # X = X.T # x의 전치행렬로 변경하여, 각 행에 2차원 좌표를 위치시킨다

cov, mean = cv2.calcCovarMatrix(X, mean = None, flags=cv2.COVAR_NORMAL+cv2.COVAR_ROWS)
# x의 각 행(cv2.COVAR_ROWS)에서 (x, y) 좌표의 평균 mean, 공부산 행렬 cov를 계산한다.
# 평균 mean은 1x2 열벡터이고, 공분산 행렬 cov는 2x2 행렬이다
# 만약 데이터가 행렬의 열에 있으면 flags에 cv2.COVAR_COLS를 사용한다
print('mean = ', mean)
print('cov=', cov)

ret, icov = cv2.invert(cov) # 공분산 행렬 cov의 역행렬 icov를 계산한다
print('icov = ', icov)

v1 = np.array([[0], [0]], dtype=np.float64)
v2 = np.array([[0], [50]], dtype=np.float64)

dist = cv2.Mahalanobis(v1, v2, icov) # 두 벡터 v1, v2 사이의 통계적 거리인 마하라노비스 거리로 공분산 행렬의 역행렬을 이용하여 계산한다
print('dist = ', dist)

cv2.waitKey()
cv2.destroyAllWindows()