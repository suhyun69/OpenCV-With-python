import cv2
import numpy as np

img = cv2.imread('../data/lena.jpg')
# lena.jpg 영상을 컬러 cv2.IMREAD_COLOR 영상으로 img에 읽는다.

print('img.ndim = ', img.ndim);
print('img.shape = ', img.shape);
print('img.dtype = ', img.dtype);
# img 영상은 img.ndim = 3으로 3차원 배열이고
# img.shape = (512, 512, 3)으로 512 x 512 크기의 3채널 영상이다.
# img.shape[0]은 영상의 세로 화소 크기, img.shape[1]은 영상의 가로 화소 크기, img.shape[2]는 영상의 채널 개수이다.
# 각 화소의 자료형은 img.dtype = uint8로 부호 없는 8비트 정수이다ㅣ

img = img.astype(np.int32) # img.astype(np.int32)은 img의 화소 자료형을 정수형으로 변경한다.
print('img.dtype = ', img.dtype)

img = np.uint8(img) # imt = np.uint8(img_)로 img의 화소 자료형을 uint8로 변경할 수 있다
print('img.dtype = ', img.dtype)