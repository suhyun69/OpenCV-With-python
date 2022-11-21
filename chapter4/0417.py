import cv2
import numpy as np

src1 = cv2.imread('../data/lena.jpg', cv2.IMREAD_GRAYSCALE)
src2 = np.zeros(shape=(512,512), dtype=np.uint8)+100

dst1 = src1 + src2 # numpy의 배열 덧셈으로 src1 + src2를 계산하면, 덧셈 결과가 255를 넘는 경우 256으로 나눈 나머지를 계산한다
dst2 = cv2.add(src1, src2) # cv2.add() 함수로 src1, src2를 덧셈하여, 덧셈 결과가 255를 넘는 경우 255로 계산한다

cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()


