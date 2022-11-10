import cv2
import numpy as np

# White 배경 생성
img = np.zeros(shape=(512, 512, 3), dtype=np.uint8) + 255
# np.ones((512, 512, 3), np.unit8) * 255
# np.full((512, 512, 3), (255, 255, 255), dtype=uint8)

# np.zeros()는 영상으로 사용할 0으로 초기화된 배열을 생성한다. shape=(512, 512, 3)은 512x512 크기의 3채널 컬러 영상,
# dtype=np.unit8은 영상 화소가 부호 없는 8비트 정수이다. 화소값이 (0, 0, 0)이면 검은색(black) 배경 영상이다
# np.zeros() + 255를 사용하면 영상의 모든 채널 값이 255로 변경되어 흰색 배경이다
# np.ones()는 1로 초기화된 배열을 생성한다. np.full()을 사용하면 배경으로 사용할 컬러를 지정하여 영상을 지정할 수 있다

pt1 = 100, 100
pt2 = 400, 400
cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)

cv2.line(img, (0,0), (500, 0), (255, 0, 0), 5)
cv2.line(img, (0,0), (0, 500), (0, 0, 255), 5)

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()