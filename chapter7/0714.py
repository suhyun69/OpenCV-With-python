import cv2
import numpy as np

#1
# 컬러 입력 영상 src를 HSV 컬러 영상 hsv로 변환한다
# 입력 영상 src 또는 hsv의 각 화소의 컬러가 data의 행에 배치되도록 모양을 변환한다.
# data.shape = (230400, 3)이다
src = cv2.imread('../data/hand.jpg')
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

data = src.reshape((-1,3)).astype(np.float32)

#2
# cv2.kmeans()로 'hand.jpg'는 K=2, 'SegmentTest.jpg'는 K=5로 클러스터링한다.
# centers.shape = (2,3)로 centers에 K개의 클러스터 중심점을 반환한다.
# labels.shap = (230400, 1)로 labels는 각 데이터 점의 클러스터 번호를 반환한다.
# ret는 클러스터 응집도를 반환한다.
K = 2
term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret, labels, centers = cv2.kmeans(data, K, None, term_crit, 5, cv2.KMEANS_RANDOM_CENTERS)
print('centers.shpae=', centers.shape)
print('labels.shape=', labels.shape)
print('ret=', ret)

#3
# centers를 np.uint8 자료형으로 변환하고, res = centers[labels.flatten()]로 res에 레이블에 대한 클러스터 중심으로 변환한 res를 생성한다.
# res.shape = (230400, 3)이다.
# res를 src와 같은 영상 모양으로 변환한다.
# 주석처리 부분은 labels2의 각 클러스터 번호에 난수로 생성한 컬러를 지정하여 dst 영상을 생성한다.
centers = np.uint8(centers)
res = centers[labels.flatten()]
dst = res.reshape(src.shape)

'''
labels2 = np.unit8(labels.reshape(src.shape[:2]))
print('labels2.max()=', labels2.max())
dst = np.zeros(src.shape, dtype=src.dtype)
for i in range(K):
    r = np.random.randint(256)
    g = np.random.randint(256)
    b = np.random.randint(256)
    dst[labels2 == i] = [b, g, r]
'''

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()