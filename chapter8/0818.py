import cv2
import numpy as np

#1
# lena.jpg를 그레이스케일로 gray에 읽는다
gray = cv2.imread('../data/lena.jpg', cv2.IMREAD_GRAYSCALE)

#2
# cv2.integral로 gray에서 적분 영상 gray_sum을 계산한다.
# cv2.normalize()로 [0, 255]로 정규화한다
# 최소값은 왼쪽 위고, 오른쪽 아래로 갈수록 값이 누적되어 최대값은 오른쪽 아래이다
gray_sum = cv2.integral(gray)
dst = cv2.normalize(gray_sum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()