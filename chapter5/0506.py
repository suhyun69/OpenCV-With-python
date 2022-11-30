import cv2
from matplotlib import pyplot as plt

bgr = cv2.imread('../data/lena.jpg')

#1
hist01 = cv2.calcHist([bgr], [0,1], None, [32, 32], [0, 256, 0, 256])
plt.title('hist01')
plt.ylim(0, 31)
plt.imshow(hist01, interpolation="nearest")
plt.show()

#2
hist02 = cv2.calcHist([bgr], [0,2], None, [32, 32], [0, 256, 0, 256])
plt.title('hist02')
plt.ylim(0, 31)
plt.imshow(hist02, interpolation="nearest")
plt.show()

#3
hist12 = cv2.calcHist([bgr], [1,2], None, [32, 32], [0, 256, 0, 256])
plt.title('hist12')
plt.ylim(0, 31)
plt.imshow(hist12, interpolation="nearest")
plt.show()