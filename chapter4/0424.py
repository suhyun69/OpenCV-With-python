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
ret, icov = cv2.invert(cov) # 공분산 행렬 cov의 역행렬 icov를 계산한다

dst = np.full((512, 512, 3), (255, 255, 255), dtype=np.uint8)
rows, cols, channels = dst.shape
centerX = cols // 2
centerY = rows // 2

v2 = np.zeros((1, 2), dtype=np.float64)

FLIP_Y = lambda y: rows - 1 - y # 좌표를 rows-1-y로 변환하여 y축을 반전시킨다

# draw Mahalanobis distance
for y in range(rows):
    for x in range(cols):
        v2[0, 0] = x - centerX
        v2[0, 1] = FLIP_Y(y) - centerY # y-축 뒤집기
        dist = cv2.Mahalanobis(mean, v2, icov)
        # rows x cols의 각 좌표를 중심점(cetnerX, centerY)을 원점으로 변환한 벡터 v2와 평균 벡터 mean의 마하라노비스 거리 dist를 계산한다 -> 색상 설정
        if dist < 0.1:
            dst[y, x] = [50, 50, 50]
        elif dist < 0.3:
            dst[y, x] = [100, 100, 100]
        elif dist < 0.8:
            dst[y, x] = [200, 200, 200]
        else:
            dst[y, x] = [250, 250, 250]

for k in range(X.shape[0]):
    x, y = X[k,:] # X의 k번째 행의 좌표(x, y)를 원점(centerX, centerY)을 기준으로 좌표(cx, cy)로 변환하고, cy = FLIP_Y(cy)로 y좌표를 반전시켜 cv2.circle() 함수로 dst에 빨간색(0, 0, 255) 원으로 표시한다
    cx = int(x + centerX)
    cy = int(y + centerY)
    cy = FLIP_Y(cy)
    cv2.circle(dst, (cx, cy), radius=5, color=(0,0,255), thickness=-1)

# draw X, Y-axes
cv2.line(dst, (0, 256), (cols-1, 256), (0,0,0))
cv2.line(dst, (256, 0), (256, rows), (0, 0 , 0))

# calculate eigen vectors
ret, eVals, eVects = cv2.eigen(cov) # 공분산 행렬 cov의 고유값 eVals, 고유 벡터 eVects를 계산한다
print('eVals = ', eVals)
print('eVects = ', eVects)

# 고유값 eVal, 고유 벡터 eVect를 이용하여 고유 벡터 위의 대칭인 두 좌표(x1, y1), (x2, y2)을 계산한다.
def ptsEigenVector(eVal, eVect):
    ## global mX, centerX, centerY
    scale = np.sqrt(eVal) # eVal[0]
    x1 = scale * eVect[0]
    y1 = scale * eVect[1]
    x2, y2 = -x1, -y1 # 대칭

    x1 += mean[0, 0] + centerX
    y1 += mean[0, 1] + centerY
    x2 += mean[0, 0] + centerX
    y2 += mean[0, 1] + centerY
    y1 = FLIP_Y(y1)
    y2 = FLIP_Y(y2)
    return int(x1), int(y1), int(x2), int(y2)

# draw eVects[0]
x1, y1, x2, y2 = ptsEigenVector(eVals[0], eVects[0])
cv2.line(dst, (x1, y1), (x2, y2), (255, 0, 0), 2)

# draw eVects[1]
x1, y1, x2, y2 = ptsEigenVector(eVals[1], eVects[1])
cv2.line(dst, (x1, y1), (x2, y2), (255, 0, 0), 2)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()


