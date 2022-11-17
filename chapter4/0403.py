import cv2

img = cv2.imread('../data/lena.jpg', cv2.IMREAD_GRAYSCALE)
img[100, 200] = 0 # 화소값(밝기, 그레이스케일) 변경
# img[100, 200] = 0은 img 영상의 y=100, x=200 화소의 값(밝기, 그레이스케일)을 0으로 변경한다.
# img[100, 200]은 img[100][200]과 같다.

print(img[100:110, 200:210]) # ROI 접근
# numpy의 슬라이싱으로 y=100에서 y=109까지, x=200에서 x=209까지의 10 x 10 사각 영역을 ROI로 지정하여 화소값을 출력한다

for y in range(100, 400):
    for x in range(200, 300):
        img[y, x] = 0
# img[100:400, 2000:300] = 0 # ROI 접근
# for 문으로 영상의 .y=100에서 y=399까지, x=200에서 x=299까지의 사각 영역을 0으로 변경한다
# numpy의 슬라이싱으로 ROI를 지정하여 img[100:400, 2000:300] = 0으로 변경할 수 있다

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()