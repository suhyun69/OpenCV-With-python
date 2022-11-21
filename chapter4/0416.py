import cv2
src = cv2.imread('../data/lena.jpg')

rows, cols, channls = src.shape

# cv2.getRotationMatrix2D(center, angel, scale) -> M
# center 좌표를 중심으로 scale 확대/축소하고 angle 각도만큼 회전한 어파인 변환행렬 M 반환
M1 = cv2.getRotationMatrix2D((rows/2, cols/2), 45, 0.5)
M2 = cv2.getRotationMatrix2D((rows/2, cols/2), -45, 1.0)

# cv2.wardAffine() 함수는 src 영상에 2x3 어파인 변환행렬 M을 적용하여 dst에 반환한다
dst1 = cv2.warpAffine(src, M1, (rows, cols))
dst2 = cv2.warpAffine(src, M2, (rows, cols))

cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()
