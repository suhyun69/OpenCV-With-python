import cv2
import numpy as np

X = np.array([[0, 0, 0, 100, 100, 150, -100, -150],
              [0, 50, -50, 0, 30, 100, -20, -100]],
             dtype=np.float64)
X = X.transpose() # X = X.T # x의 전치행렬로 변경하여, 각 행에 2차원 좌표를 위치시킨다

mean, eVects = cv2.PCACompute(X, mean = None) # X의 평균 벡터 mean, 공분산 행렬의 고유 벡터 eVects를 계산한다 = 0424.py
print('mean = ', mean)
print('eVect = ', eVects)

Y = cv2.PCAProject(X, mean, eVects) # cv2.PCAProject() 함수는 고유 벡터 eVects에 의해 PCA 투영한다. 즉, 데이터르 고유 벡터를 축으로 한 좌표로 변환한다.
print('Y = ', Y)


X2 = cv2.PCABackProject(Y, mean, eVects) # Y를 PCA 역투영하면 원본 X를 복구할 수 있다.
print('X2 = ', X2)
print(np.allclose(X, X2)) # X와 X2는 오차 범위 내에서 같은 값을 갖는다. True.
cv2.waitKey()
cv2.destroyAllWindows()