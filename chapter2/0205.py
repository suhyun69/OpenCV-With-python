import cv2
from matplotlib import pyplot as plt

imageFile = '../data/lena.jpg'
imgGray = cv2.imread(imageFile, cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(6,6)) # 크기를 (6인치, 6인치)로 설정한다

plt.subplots_adjust(left=0, right=1, bottom=0, top=1) # 영상 출력 범위를 좌우를 [0,1], 위아래를 [0,1]로 조정한다. 범위는 left < right와 bottom < top이어야 한다
plt.imshow(imgGray, cmap='gray')

plt.axis('off')
plt.show()