import cv2
from matplotlib import pyplot as plt

# src.shape = (128, 64, 3)의 컬러 영상 'people1.png'를 src에 로드하고, 그래디언트의 크기 matgnituede와 방향 direction을 계산하여 HOG 디스크립터를 계산한다
src = cv2.imread('../data/people1.png')

#1: HOG in color image
# cv2.HOGDescriptor()로 기본값으로 객체 hog1을 생성하고, hog1.compute()로 영상 src에서 HOG 디스크립터 des1를 계산한다.
# des1.shape = (3780,)이다. hog1.getDescriptorSize()는 디스크립터의 크기(3780)을 반환한다
hog1 = cv2.HOGDescriptor()
des1 = hog1.compute(src)
print("HOG feature size = ", hog1.getDescriptorSize())
print('des1.shape=', des1.shape)

#2: HOG in coloir image
# 기본값을 설명하기 위하여 변수를 초기화하고 계산한다.
# hog2 객체를 생성하고, hog2.compute()로 영상 src에서 HOG 디스크립터 des12를 계산한다.
# des2.shape = (3780,)이다. des1과 des2는 결과가 같다
hog2 = cv2.HOGDescriptor(_winSize=(64, 128),
                         _blockSize=(16, 16),
                         _blockStride=(8, 8),
                         _cellSize=(8, 8),
                         _nbins=9,
                         _derivAperture=1,
                         _winSigma=-1,
                         _histogramNormType=0,
                         _L2HysThreshold=0.2,
                         _gammaCorrection=True,
                         _nlevels=64,
                         _signedGradient=False)

des2 = hog2.compute(src)
print('des2.shape=', des2.shape)

#3:
# _winSize=(64, 128), _blockSize=(16, 16), _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9를 설정하여 hog3 객체를 생성하고, HOG 디스크립터 des3를 계산한다
# des3.shape=3780,)이다. hog3.gammaCorrection = False이기 때문에 des3은 des1, des2와 약간 다르다
hog3 = cv2.HOGDescriptor(_winSize=(64, 128),
                          _blockSize=(16, 16),
                          _blockStride=(8, 8),
                          _cellSize=(8, 8),
                          _nbins=9)

des3 = hog3.compute(src)
print('des3.shape=', des3.shape)

#4 HOG in grayscale image
# 그레이스케일 영상 gray로 변경하여 hog3.compute(gray)로 HOG 디스크립터 des4를 계산한다.
# des4.shape=(3780,)이고, 에지 그래디언트 계산에서 차이로 인하여 디스크립터 값은 약간 다른 값을 찾는다
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
des4 = hog3.compute(gray)
print('des4.shape=', des4.shape)

#5
# matyplotlib를 사용하여 des1, des2, des3, des4를 36개씩 건너뛰어 샘플링하여 그래프로 표시한다.
# des1, des2는 같고, des3, des4는 약간 다른 값을 갖느다
plt.title('HOGDescriptor')
plt.plot(des1[::36], color='b', linewidth=4, label='des1')
plt.plot(des2[::36], color='g', linewidth=4, label='des2')
plt.plot(des3[::36], color='r', linewidth=2, label='des3')
plt.plot(des4[::36], color='y', linewidth=1, label='des4')
plt.legend(loc='best')
plt.show()

'''
(3780,)의 디스크립터를 계산하는 과정은 다음과 같다
셀 크기 cellSize=(8,8)은 검은색 사각형에서 그래디언트를 계산한다
src 영상은 가로64//8 = 8, 세로 128//8 = 16으로 (8x16) = 128개의 셀로 나누어진다 

_signedGradient=False, _nbins=9에 의해 [0, 180]도 범위를 9개 빈(0, 20, 40, 60, 80, 100, 120, 140, 160)에 그래디언트의 크기 magnitude를 누적시켜 히스토그램을 계산한다
방향에 대한 정확한 빈이 없는 경우 크기를 비례하여 분배한다
즉, 각 셀에서 9x1 벡터를 계산한다

다음은 발간색 블록 크기 _blockSize=(16, 16) 화소에 의해 4개의 셀의 히스토그램을 묶어 36x1 벡터를 생성하고, _histogramNormType에 따라 정규화한다

_blockSize=(16, 16)의 블록을 전체 영상에서 _blockStride=(8,8)로 움직이면 가로로 7번, 세로로 15번 이동하여 7x15 = 105개의 블록에서 각각 36x1 벡터를 계산한다
그러므로 전체 디스크립터는 105x36=3780 크기의 벡터로 계산한다. 
디스크립터가 블록 단위로 정규화되어 있어,
cv2.norm(des1[:36]),
cv2.norm(des2[:36]),
cv2.norm(des3[:36]),
cv2.norm(des1[36:72]),
cv2.norm(des2[36:72]),
cv2.norm(des3[36:72])
등이 1에 가까운 값이다. 오차를 더하여 정규화한다
'''