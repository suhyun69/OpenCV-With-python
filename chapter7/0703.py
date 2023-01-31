import cv2
import numpy as np

src = cv2.imread('../data/rect.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 100)

# edges에서, cv2.HoughLinesP()로 rho = 1, tehta = np.pi / 180.0, threshold = 100을 적용하여 선분을 lines에 검출한다.
# 검출된 직선의 rho, theta를 저장한 lines 배열의 모양은 lines.shape = (4, 1, 4)이다.
# 4개의 직선의 rho, theta를 저장한 (1, 4)로 이해하면 된다
lines = cv2.HoughLinesP(edges, rho = 1, theta = np.pi / 180.0, threshold = 100)
print('lines.shape=', lines.shape)

# for 문에서 각 선분의 양 끝점 정보는 x1, y1, x2, y2 = line[0]이고, cv2.lines()으로 src에 직선을 그린다
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(src, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow('edges', edges)
cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()