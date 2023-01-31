import cv2
import numpy as np

src = cv2.imread('../data/rect.jpg')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 100)

# edges에서, cv2.HoughLines()로 rho = 1, tehta = np.pi / 180.0, threshold = 100을 적용하여 직선을 lines에 검출한다.
# 검출된 직선의 rho, theta를 저장한 lines 배열의 모양은 lines.shape = (4, 1, 2)이다.
# 4개의 직선의 rho, theta를 저장한 (1, 2)로 이해하면 된다
lines = cv2.HoughLines(edges, rho = 1, theta = np.pi / 180.0, threshold = 100)
print('lines.shape=', lines.shape)

# for 문에서 각 직선의 매개변수는 rho, theta = line[0]이고, rho, theta를 이용하여 검출된 직선을 그린다.
# 원점에서 (rho, theta)에 의한 직선과 직각으로 만나는 좌표 (x0, y0)는 xo = rho * c, y0 = rho * s로 계산한다.
# 직선 방향으로의 단위 벡터는 (cos(theta), -sin(tehta))이다.
# 이 단위 벡터를 +, -방향으로 스케일링하고, x0, y0에 더하여 선분의 양 끝점 (x1, y1)과 (x2, y2)를 계산하여 cv2.line()으로 src에 직선을 그리면 4개의 직선이 표시된다.
for line in lines:
    rho, theta = line[0]
    c = np.cos(theta)
    s = np.sin(theta)
    x0 = c * rho
    y0 = s * rho
    x1 = int(x0 + 1000 * (-s))
    y1 = int(y0 + 1000 * (c))
    x2 = int(x0 - 1000 * (-s))
    y2 = int(y0 - 1000 * (c))
    cv2.line(src, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow('edges', edges)
cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()