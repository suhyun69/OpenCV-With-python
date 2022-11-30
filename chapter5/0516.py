import cv2
import numpy as np

#1
# 원본 영상 tsukuba_l.png 파일을 src에 읽고 표시한다
# 대부분의 배경이 어둡고 조각상의 얼굴만 밝은 영상이다
src = cv2.imread('../data/tsukuba_l.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('src', src)

#2
# cv2.equalizeHist()로 전체 영상에 대해 하나의 히스토그램을 이용하여 dst에 평활화한 결과이다
# 영상 전체가 밝아져서, 얼굴 부분의 윤곽선이 구분되지 않는다
dst = cv2.equalizeHist(src)
cv2.imshow('dst', dst)

#3
# cv2.createCLAHE(clipLimit=40, tileGridSize=(1,1))로 하나의 히스토그램만을 가지고 dst2에 CLAHE 히스토그램 평활화한다.
# 하나의 히스토그램을 사용하기 때문에 dst와 비슷한 결과를 갖는다
# dst와 dst2는 정확히 같지 않다
# CLAHE는 히스토그램 재분배를 수행하고, 변환 테이블 계산 방법이 cv2.equalizeHist()와 다르기 때문이다
# src 영상 전체의 히스토그램은 clipLimit = 40 * src.size / 256 = 17280.0보다 큰 빈 값이 없기 때문에 재분배는 수행하지 않는다
clahe2 = cv2.createCLAHE(clipLimit=40, tileGridSize=(1,1))
dst2 = clahe2.apply(src)
cv2.imshow('dst2', dst2)

#4
# cv2.createCLAHE(clipLimit=40, tileGridSize=(8,8))로 tileGridSize=(8,8)개의 타일로 나누어 dst3에 CLAHE 히스토그램 평활화한다.
# dst3은 배경뿐만 아니라 얼굴 부분에서 dst, dst2에 비해 대비가 선명한 영상을 얻는다
clahe3 = cv2.createCLAHE(clipLimit=40, tileGridSize=(8,8))
dst3 = clahe3.apply(src)
cv2.imshow('dst3', dst3)

cv2.waitKey()
cv2.destroyAllWindows()