import cv2
from matplotlib import pyplot as plt

imageFile = '../data/lena.jpg'
imgBGR = cv2.imread(imageFile)
plt.axis('off') # X, Y축을 표시하지 않는다

# OpenCV로 읽은 컬러 영상 imgBGR의 채널 순서 BGR을 cvtColor()로 RGB 채널 순서로 변경한다.
# OpenCV는 컬러 영상을 BGR 채널 순서로 처리하고, Matplotlib는 RGB 채널 순서로 처리하기 때문이다
imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)

plt.imshow(imgRGB)
plt.show()