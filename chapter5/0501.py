import cv2
import numpy as np
src = cv2.imread('../data/heart10.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('src', src)

# src에서 thresh=120, max_val=255, type=cv2.THRESH_BINARY로 임계값 적용하여 이진 영상 dst 생성
ret, dst = cv2.threshold(src, 120, 255, cv2.THRESH_BINARY)
print('ret=', ret)
cv2.imshow('dst', dst)

# src에서 thresh=200, max_val=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU로 임계값 적용하여 이진 영상 dst2 생성
# 주어진 임계값 200과 상관없이 Otsu 알고리즘으로 최적 임계값을 ret2=175로 계산한다
ret2, dst2 = cv2.threshold(src, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print('ret2=', ret2)
cv2.imshow('dst2', dst2)

cv2.waitKey()
cv2.destroyAllWindows()

