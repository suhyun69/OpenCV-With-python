import cv2
import numpy as np

src = cv2.imread('../data/morphology.jpg', cv2.IMREAD_GRAYSCALE)

# shape = cv2.MORPH_RECT, ksize=(3,3)으로 사각형 kernel을 생성하고, cv2.erode(), cv2.dilate() 함수로 모폴로지 연산을 수행한다
kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3,3))

# iterations = 5로 다섯 번 침식 연산한다. 흰색의 침식으로 검은색 속의 흰색 점은 사라지고, 흰색 물체 속의 검은색 점은 커진다
erode = cv2.erode(src, kernel, iterations = 5)

# iterations = 5로 다섯 번 팽창 연산한다. 흰색의 팽창으로 검은색 속의 흰색 점은 더 커지고, 흰색 물체 속의 검은색 점은 채워진다.
dilate = cv2.dilate(src, kernel, iterations = 5)

# 팽창으로 물체 내부의 구멍을 채운 dilate 영상에 cv2.erode()로 iterations = 7로 일곱 번 침식 연산한다.
# 흰색의 침식으로 검은색 속의 흰색 잡음을 제거한다.
# 흰색 물체를 src의 크기로 되돌리려면, erode2를 iterations = 2로 두 번 팽창하면 된다.
erode2 = cv2.erode(dilate, kernel, iterations = 7)

cv2.imshow('src', src)
cv2.imshow('erode', erode)
cv2.imshow('dilate', dilate)
cv2.imshow('erode2', erode2)
cv2.waitKey()
cv2.destroyAllWindows()

