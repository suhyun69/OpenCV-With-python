import cv2

img = cv2.imread('../data/lena.jpg')

img[100:400, 200:300, 0] = 255 # B-채널을 255로 변경
img[100:400, 200:300, 1] = 255 # G-채널을 255로 변경
img[100:400, 200:300, 2] = 255 # R-채널을 255로 변경

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()