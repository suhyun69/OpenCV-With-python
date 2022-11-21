import cv2
src = cv2.imread('../data/lena.jpg', cv2.IMREAD_GRAYSCALE)

# dst = src # 참조. dst가 src를 참조하여, dst를 변경하면 src도 변경됨에 주의한다
dst = src.copy() # 복사
dst[100:400, 200:300] = 0

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()