import cv2
import numpy as np

# 입력 A의 골격화(skeleton)를 구조요소 B를 사용한 침식erosion과 열기opening에 의한 골격화skeleton를 구현한다
# 침식 연산이 공집합일 때까지 합집합을 계산한다

#1
# 입력 영상 src에 임계값을 적용하여 이진 영상 A를 생성하고, 결과를 위한 skel_dst를 생성한다
src = cv2.imread('../data/T.jpg', cv2.IMREAD_GRAYSCALE)
# src = cv2.imread('../data/alphabet.bmp', cv2.IMREAD_GRAYSCALE)

ret, A = cv2.threshold(src, 128, 255, cv2.THRESH_BINARY)
skel_dst = np.zeros(src.shape, np.uint8)

# 구조요소 B를 shape1 = cv2.MORPH_CROSS 또는 shape2 = cv2.MORPH_RECT로 생성한다
# A를 B로 erode 침식하고, erode를 B로 opening 열기한 다음, tmp = erode-opening을 계산한 뒤에 cv2.bitwise_or() 비트 연산으로 skel_dst에 합집합을 계산한다.
# 다음 반복을 위해 erode를 A에 복사하고, cv2.countNonZero(A)를 이용하여 A가 공집합이 아니면 계속 반복한다.
shape1 = cv2.MORPH_CROSS
# shape1 = cv2.MORPH_RECT
B = cv2.getStructuringElement(shape = shape1, ksize=(3,3))
done = True
while done:
    erode =cv2.erode(A, B)
    opening = cv2.morphologyEx(erode, cv2.MORPH_OPEN, B)
    tmp = cv2.subtract(erode, opening)
    skel_dst = cv2.bitwise_or(skel_dst, tmp)
    A = erode.copy()
    done = cv2.countNonZero(A) != 0

cv2.imshow('src', src)
cv2.imshow('skel_dst', skel_dst)
cv2.waitKey()
cv2.destroyAllWindows()