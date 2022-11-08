import cv2
from matplotlib import pyplot as plt

imageFile = '../data/lena.jpg'
imgGray = cv2.imread(imageFile, cv2.IMREAD_GRAYSCALE)
plt.axis('off') # X, Y축을 표시하지 않는다

plt.imshow(imgGray, cmap='gray', interpolation='bicubic') # imgGray 영상을 'gray' 컬러맵, 'bicubic'으로 보간한다
plt.show()