import cv2
import numpy as np

#1
# 512x512 크기의 배경이 (255, 255, 255)인 3-채널 컬러 영상 src를 생성하고 사각형과 원을 그린다.
src = np.full((512, 512, 3), (255, 255, 255), dtype = np.uint8)
cv2.rectangle(src, (50, 50), (200, 200), (0, 0, 255), 2)
cv2.circle(src, (300, 300), 100, (0,0, 255), 2)

#2
# src를 dst에 복사하고, cv2.floodFill()로 dst의 seedPoint = (100, 100)을 시작점으로 사각형 내부를 newVal = (255, 0, 0) 색상으로 dst에 채운다.
dst = src.copy()
cv2.floodFill(dst, mask = None, seedPoint = (100, 100), newVal = (255, 0, 0))

#3
# cv2.floodFill()로 dst의 seedPoint = (300, 300)를 시작점으로 원의 내부를 newVal = (0, 255, 0) 색상으로 dst에 채운다.
# 원의 내부를 채운 영역의 바우딩 사각형 rect를 이용하여 dst2에 사각형을 그린다.
retval, dst2, mask, rect = cv2.floodFill(dst, mask = None, seedPoint = (300, 300), newVal = (0, 255, 0))

print('rect=', rect)
x, y, width, height = rect
cv2.rectangle(dst2, (x, y), (x + width, y + height), (255, 0, 0), 2)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()