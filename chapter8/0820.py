import cv2
import numpy as np

#1
# 적분 영상 sumImag을 이용하여 5가지 형태의 다양한 크기에서 모든 가능한 특징을 계산하여, [특징번호, x, y, w, h, 특징값]의 항목을 갖는 리스트로 계산하여 반환한다
def rectSum(sumImage, rect):
    x, y, w, h = rect
    a = sumImage[y, x]
    b = sumImage[y, x + w]
    c = sumImage[y + h, x]
    d = sumImage[y + h, x + w]
    return a + d - b - c

def compute_Haar_feature1(sumImage):
    rows, cols = sumImage.shape
    rows -= 1
    cols -= 1
    f1 = []
    for y in range(0, rows):
        for x in range(0, cols):
            for h in range(1, rows - y + 1):
                for w in range(1, (cols - x) // 2 + 1):
                    s1 = rectSum(sumImage, (x, y, w, h))
                    s2 = rectSum(sumImage, (x + w, y, w, h))
                    f1.append([1, x, y, w, h, s1 - s2])
    return f1

def compute_Haar_feature2(sumImage):
    rows, cols = sumImage.shape
    rows -= 1
    cols -= 1
    f2 = []
    for y in range(0, rows):
        for x in range(0, cols):
            for h in range(1, (rows - y) // 2 + 1):
                for w in range(1, cols - x + 1):
                    s1 = rectSum(sumImage, (x, y, w, h))
                    s2 = rectSum(sumImage, (x, y + h, w, h))
                    f2.append([2, x, y, w, h, s2 - s1])
    return f2

def compute_Haar_feature3(sumImage):
    rows, cols = sumImage.shape
    rows -= 1
    cols -= 1
    f3 = []
    for y in range(0, rows):
        for x in range(0, cols):
            for h in range(1, rows - y + 1):
                for w in range(1, (cols - x) // 3 + 1):
                    s1 = rectSum(sumImage, (x, y, w, h))
                    s2 = rectSum(sumImage, (x + w, y, w, h))
                    s3 = rectSum(sumImage, (x + 2 * w, y, w, h))
                    f3.append([3, x, y, w, h, s1 - s2 + s3])
    return f3

def compute_Haar_feature4(sumImage):
    rows, cols = sumImage.shape
    rows -= 1
    cols -= 1
    f4 = []
    for y in range(0, rows):
        for x in range(0, cols):
            for h in range(1, (rows - y) // 3 + 1):
                for w in range(1, cols - x + 1):
                    s1 = rectSum(sumImage, (x, y, w, h))
                    s2 = rectSum(sumImage, (x, y + h, w, h))
                    s3 = rectSum(sumImage, (x, y + 2 * h, w, h))
                    f4.append([4, x, y, w, h, s1 - s2 + s3])
    return f4

def compute_Haar_feature5(sumImage):
    rows, cols = sumImage.shape
    rows -= 1
    cols -= 1
    f5 = []
    for y in range(0, rows):
        for x in range(0, cols):
            for h in range(1, (rows - y) // 2 + 1):
                for w in range(1, (cols - x) // 2 + 1):
                    s1 = rectSum(sumImage, (x, y, w, h))
                    s2 = rectSum(sumImage, (x + w, y, w, h))
                    s3 = rectSum(sumImage, (x, y + h, w, h))
                    s4 = rectSum(sumImage, (x + w, y + h, w, h))
                    f5.append([5, x, y, w, h, s1 - s2 - s3 + s4])
    return f5

#2
# 24 x 24 영상 'lenaFace24.jpg'를 gray에 입력하고, cv2.integral()로 적분 영상 gray_sum을 계산한다.
# compute_Haar_feature1()로 모든 가능한 에지 특징을 f1에 계산한다
# f1은 len(f1) = 43200개의 특징을 갖고 있다. 
# 첫 특징 f1[0]=[1, 0, 0,1, 1, -11]에서 형태 특징 x = 0, y = 0, w = 1, h = 1의 특징 -11을 표현한다
gray = cv2.imread('../data/lenaFace24.jpg', cv2.IMREAD_GRAYSCALE)
gray_sum = cv2.integral(gray)
f1 = compute_Haar_feature1(gray_sum)
n1 = len(f1)
print('len(f1)=', n1)
for i, a in enumerate(f1[:2]):
    print('f1[{}]={}'.format(i, a))

#3
# compute_Haar_feature2()로 모든 가능한 에지 특징을 f2에 계산한다
# f2은 len(f2) = 43200개의 특징을 갖고 있다. 
# 첫 특징 f2[0]=[2, 0, 0,1, 1, 25]에서 형태 특징 x = 0, y = 0, w = 1, h = 1의 특징 25을 표현한다
f2 = compute_Haar_feature2(gray_sum)
n2 = len(f2)
print('len(f2)=', n2)
for i, a in enumerate(f2[:2]):
    print('f2[{}]={}'.format(i, a))

#4
# compute_Haar_feature3()로 모든 가능한 에지 특징을 f3에 계산한다
# f3은 len(f3) = 27600 특징을 갖고 있다. 
# 첫 특징 f3[0]=[3, 0, 0,1, 1, 138]에서 형태 특징 x = 0, y = 0, w = 1, h = 1의 특징 138을 표현한다
f3 = compute_Haar_feature3(gray_sum)
n3 = len(f3)
print('len(f3)=', n3)
for i, a in enumerate(f3[:2]):
    print('f3[{}]={}'.format(i, a))

#5
# compute_Haar_feature4()로 모든 가능한 에지 특징을 f4에 계산한다
# f4은 len(f4) = 27600 특징을 갖고 있다. 
# 첫 특징 f4[0]=[4, 0, 0,1, 1, 170]에서 형태 특징 x = 0, y = 0, w = 1, h = 1의 특징 170을 표현한다
f4 = compute_Haar_feature4(gray_sum)
n4 = len(f4)
print('len(f4)=', n4)
for i, a in enumerate(f4[:2]):
    print('f4[{}]={}'.format(i, a))

# 6
# compute_Haar_feature5()로 모든 가능한 에지 특징을 f5에 계산한다
# f5은 len(f5) = 20736 특징을 갖고 있다. 
# 첫 특징 f5[0]=[5, 0, 0,1, 1, -44]에서 형태 특징 x = 0, y = 0, w = 1, h = 1의 특징 -44을 표현한다
f5 = compute_Haar_feature5(gray_sum)
n5 = len(f5)
print('len(f5)=', n5)
for i, a in enumerate(f5[:2]):
    print('f5[{}]={}'.format(i, a))

#7
# 24 x 24 영상에서 가능한 전체 특징은 total features = 162336개이다. 
# 화소의 개수 24 x 24 = 576보다 훨씬 많은 특징이 계산된다
# 이런한 특징이 모두 유효하고 물체 인식을 위해 중요한 특징은 아니다.
# Viola와 Jones는 얼굴 검출을 위해 Adabootst와 캐스캐이드 분류기를 이용하여 약 6,000개의 특징을 찾아서 얼굴 영역을 검출한다
print('total features = ', n1 + n2 + n3 + n4 + n5)