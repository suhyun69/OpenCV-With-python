import cv2

img = cv2.imread('../data/lena.jpg')
img[100, 200] = [255, 0, 0]
print(img[100, 200:210])

img[100:400, 200:300] = [255, 0, 0] # ROI 접근, 파란색(blue)로 변경. 컬러는 BGR-채널 순서이다.

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()
