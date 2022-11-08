import cv2

imageFile ='../data/lena.jpg'
img = cv2.imread(imageFile)

cv2.imwrite('../data/Lena.bmp', img)
cv2.imwrite('../data/Lena.png', img)
cv2.imwrite('../data/Lena.png', img)
cv2.imwrite('../data/Lena2.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 9]) # img를 압출률 9의 PNG 영상으로 Lena2.png 파일에 저장한다. 압욱률은 [0,9]이며 압축률이 높을수록 시간이 많이 걸린다. 디폴트는 3이다
cv2.imwrite('../data/Lena2.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90]) # img를 90%의 품질을 갖는 JPEG 영상으로 Lena2.jpg 파일에 저장한다. 품질의 범위는 [0, 100]이며 높을수록 영상의 품질이 좋다. 디폴트는 95이다.