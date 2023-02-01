import cv2
import numpy as np

#1
# down2는 입력 영상 src릐 가로, 세로 각각 1/2배로 축소한 피라미드 영상이다.
# down2.shape = (256, 256, 3)이다.
# down4는 down2를 가로, 세로 각각 1/2로 축소한 피라미드 영상으로 down2.shape = (256, 256, 3)이다.
src = cv2.imread('../data/lena.jpg')

down2 = cv2.pyrDown(src)
down4 = cv2.pyrDown(down2)
print('down2.shape=', down2.shape)
print('down4.shape=', down4.shape)

#2
# up은 입력 영상 src를 가로, 세로 각각 2배로 확대한 피라미드 영상으로 up2.shae = (1024, 1024,3)이다.
# up4는 up2를 가로와 세로 각각 2배로 확대한 피라미드 영상으로 up4.shape = (2048, 2048, 3)이다
up2 = cv2.pyrUp(src)
up4 = cv2.pyrUp(up2)
print('up2.shape=', up2.shape)
print('up4.shape=', up4.shape)

cv2.imshow('down2', down2)
cv2.imshow('up2', up2)

cv2.waitKey()
cv2.destroyAllWindows()