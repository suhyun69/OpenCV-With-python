import cv2
import numpy as np

src = np.array([[0,0,0,0],
                [1,1,3,5],
                [6,1,1,3],
                [4,3,1,7]
                ], dtype=np.uint8)

hist = cv2.calcHist(images=[src], channels=[0], mask=None, histSize=[4], ranges=[0,8])
print('hist = ', hist)

backP = cv2.calcBackProject([src], [0], hist, [0,8], scale=1)
print('backP = ', backP)

'''
hist =  [[9.]
 [3.]
 [2.]
 [2.]]
backP =  [[9 9 9 9]
 [9 9 3 2]
 [2 9 9 3]
 [2 3 9 2]]
 
hist[0][0] = 9는 src에서 0과 1의 카운트이다
hist[1][0] = 3는 src에서 2와 3의 카운트이다
hist[2][0] = 2는 src에서 4와 5의 카운트이다
hist[3][0] = 2는 src에서 6과 7의 카운트이다

src(x,y)의 0,1 위치는 backP(x,y)에서 hist[0][0] = 9이다
src(x,y)의 2,3 위치는 backP(x,y)에서 hist[1][0] = 3이다
src(x,y)의 4,5 위치는 backP(x,y)에서 hist[2][0] = 2이다
src(x,y)의 6,7 위치는 backP(x,y)에서 hist[3][0] = 2이다
'''