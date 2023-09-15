import cv2
import numpy as np

#1
# 1에서 16까지 초기화된 np.uint8 자료형의 4x4 배열 A를 생성하고 출력한다
A = np.arange(1, 17).reshape(4, 4).astype(np.uint8)
print('A=', A)

#2
# cv2.integral3()로 배열 A에서 적분 영상 sumA, sqsumA, tiltedA를 계산한다
# np.unit32(sqsumA)으로 자료형을 정수로 변경하여 출력한다
sumA, sqsumA, tiltedA = cv2.integral3(A)
print('sumA=', sumA)
print('sqsumA=', np.uint32(sqsumA))
print('tiltedA=', tiltedA)