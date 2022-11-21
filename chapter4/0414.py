import cv2
import numpy as np
src = cv2.imread('../data/lena.jpg', cv2.IMREAD_GRAYSCALE)

dst = cv2.resize(src, dsize=(320,240)) # src를 가로 320, 세로 240 크기로 변환하여 dst에 저장한다
dst2 = cv2.resize(src, dsize=(0,0), fx=1.5, fy=1.2) # src를 가로 1.5배, 세로 1.2배로 변환하여 dst2에 저장한다

cv2.imshow('dst', dst)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()