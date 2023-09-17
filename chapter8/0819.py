import cv2
import numpy as np

#1
# rectSum()은 적분 영상 sumImage에서 사각형 rect의 합계를 계산한다
def rectSum(sumImage, rect):
    x, y, w, h = rect
    a = sumImage[y, x]
    b = sumImage[y, x + w]
    c = sumImage[y + h, x]
    d = sumImage[y + h, x + w]
    return a + d - b - c

#2
# 적분 영상 sumImage을 이용하여 각 특징을 계산한다
def compute_Haar_feature1(sumImage, rect):
    x, y, w, h = rect
    s1 = rectSum(sumImage, (x, y, w, h))
    s2 = rectSum(sumImage, (x + w, y, w,  h))
    return s1 - s2

def compute_Haar_feature2(sumImage, rect):
    x, y, w, h = rect
    s1 = rectSum(sumImage, (x, y, w, h))
    s2 = rectSum(sumImage, (x, y + h, w, h))
    return s2 - s1

def compute_Haar_feature3(sumImage, rect):
    x, y, w, h = rect
    s1 = rectSum(sumImage, (x, y, w, h))
    s2 = rectSum(sumImage, (x + w, y, w, h))
    s3 = rectSum(sumImage, (x + 2 * w, y, w, h))
    return s1 -s2 + s3

def compute_Haar_feature4(sumImage, rect):
    x, y, w, h = rect
    s1 = rectSum(sumImage, (x, y, w, h))
    s2 = rectSum(sumImage, (x, y + h, w, h))
    s3 = rectSum(sumImage, (x, y + 2 * h, w, h))
    return s1 - s2 + s3

def compute_Haar_feature5(sumImage, rect):
    x, y, w, h = rect
    s1 = rectSum(sumImage, (x, y, w, h))
    s2 = rectSum(sumImage, (x + w, y, w, h))
    s3 = rectSum(sumImage, (x, y + h, w, h))
    s4 = rectSum(sumImage, (x + w, y + h, w, h))
    return s1 + s4 - s2 - s3

#3
# 1에서 36까지 초기화된 6x6 배열 A를 생성하고, cv2.integral()로 적분을 sumA에 계산한다
# sumA.shape(7, 7)이다. A의 크기를 h, w에 저장한다
A = np.arange(1, 6 * 6 + 1).reshape(6, 6).astype(np.uint8)
print('A=', A)

h, w = A.shape
sumA = cv2.integral(A)
print('sumA=', sumA)

#4
f1 = compute_Haar_feature1(sumA, (0, 0, w // 2, h))
print('f1=', f1)

#5
f2 = compute_Haar_feature2(sumA, (0, 0, w, h // 2))
print('f2=', f2)

#6
f3 = compute_Haar_feature3(sumA, (0, 0, w // 3, h))
print('f3=', f3)

#7
f4 = compute_Haar_feature4(sumA, (0, 0, w, h // 3))
print('f4=', f4)

#8
f5 = compute_Haar_feature5(sumA, (0, 0, w // 2, h // 2))
print('f5=', f5)
