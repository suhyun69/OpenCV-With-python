import cv2

img = cv2.imread('../data/lena.jpg', cv2.IMREAD_GRAYSCALE) # lena.jpg 영상을 그레이스케일 영상으로 img에 읽는다
print('img.shape = ', img.shape)

img = img.flatten() # 다차원 배열을 1차원 배열로 변경하여 img.shape = (262144, )이다
print('img.shape = ', img.shape)

img = img.reshape(-1, 512, 512)
# 3차원 배열로 확장한다. -1로 표시된 부분은 크리르 자동으로 계산한다. img의 화소 크기가 512 x 512이므로 img.shape(1, 512, 512)로 변경된다.
# img.reshape()은 실제 데이터를 변경하지는 않고, 모양을 변경한다
print('img.shape = ', img.shape)

cv2.imshow('img', img[0]) # img[0].shape은 (512, 512)이다. cv2.imshow('img', img[0])은 원본 영상을 표시한다
cv2.waitKey()
cv2.destroyAllWindows()