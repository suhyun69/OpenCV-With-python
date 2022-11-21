import cv2
src = cv2.imread('../data/lena.jpg')

dst1 = cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE) # 시계방향 90도
dst2 = cv2.rotate(src, cv2.ROTATE_90_COUNTERCLOCKWISE) # 반시계방향 90도

cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()