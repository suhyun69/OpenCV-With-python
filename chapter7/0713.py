import cv2
import numpy as np

#1
# floodFillPostProcess()는 cv2.floodFill() 함수를 사용하여 src를 복사한 img영상에서 유사한 영역을 채워 분할한다.
# mask는 img보다 가로, 세로로 2만큼 큰 영상이고, 0인 화소 (x, y)를 찾아 cv2.floodFill()로 채우면, (x,y)의 화소값과 위아래로 diff 차이가 나지 않으면 img의 해당 화소는 newVal로 채우고, mask는 1로 채운다.
def floodFillPostProcess(src, diff = (2, 2, 2)):
    img = src.copy()
    rows, cols = img.shape[:2]
    mask = np.zeros(shape = (rows+2, cols+2), dtype=np.uint8)
    for y in range(rows):
        for x in range(cols):
            if mask[y+1, x+1] == 0:
                r = np.random.randint(256)
                g = np.random.randint(256)
                b = np.random.randint(256)
                cv2.floodFill(img, mask, (x, y), (b, g, r), diff, diff)
    return img

#2
# BGR 입력 영상 src를 HSV 영상 hsv로 변환한다.
# floodFillPOstPoricess()를 src, hsv에 적용하여 각각 dst, dst2로 영역 분할한다.
# 필터링하지 않은 src, hsv 영상에서 디폴트 차이 diff = (2, 2, 2)에 의한 채우기로 영역 분할한 결과는 매우 많은 영역이 검출된다.
src = cv2.imread('../data/flower.jpg')
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
dst = floodFillPostProcess(src)
dst2 = floodFillPostProcess(hsv)
cv2.imshow('src', src)
cv2.imshow('hsv', hsv)
cv2.imshow('dst', dst)
cv2.imshow('dst2', dst2)

#3
# cv2.pyrMEanShiftFiltering()로 src를 sp=5, sr=20, maxLevel=4로 피라미드 평균 이동 필터링하여 res에 저장한다.
# floodFillPOstProcess()를 res에 적용하여 dst3으로 영역 분할한다.
res = cv2.pyrMeanShiftFiltering(src, sp=5, sr=20, maxLevel=4)
dst3 = floodFillPostProcess(res)

#4
# cv2.pyrMeanShiftFiltering()으로 hsv를 sp=5, sr=20, maxLevel=4, 최대반복 횟수 10, 오차 2의 종료 조건을 적용하여 피라미드 평균 이동 필터링하여 res2에 저장한다.
# floodFillPOstProcess()를 res2에 적용하여 dst4으로 영역 분할한다.
term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 2)
res2= cv2.pyrMeanShiftFiltering(hsv, sp=5, sr=20, maxLevel=4, termcrit=term_crit)
dst4 = floodFillPostProcess(res2)

cv2.imshow('res', res)
cv2.imshow('res2', res2)
cv2.imshow('dst3', dst3)
cv2.imshow('dst4', dst4)
cv2.waitKey()
cv2.destroyAllWindows()
