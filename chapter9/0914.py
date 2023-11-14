import cv2
import numpy as np

# 1
# SIFT 객체 siftF를 생성하고, BFMatcher 또는 FlannBasedMatcher()로 매칭객체를 생성한다.
# radiusMatch()로 매칭을 계산한다
src1 = cv2.imread('../data/book1.jpg')
src2 = cv2.imread('../data/book2.jpg')
img1 = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)

# 2
# SIFT 객체 siftF를 생성하고,siftF.detectAndCompute()로 img1, img2의 각각의 특징점 kp1, kp를 검출하고, 디스크립터 des1, des2를 계산한다
# maxDistance = 50이면 len(kp1) = 597, len(kp2)=881개의 특징점을 검출한다
siftF = cv2.SIFT_create()
kp1, des1 = siftF.detectAndCompute(img1, None)
kp2, des2 = siftF.detectAndCompute(img2, None)
print('len(kp1)=', len(kp1))
print('len(kp2)=', len(kp2))

# 3
# BFMNatcher 객체 bf를 생성하고 bf.radiusMatch()로 매칭을 계산하거나, FlannBasedMatcher 객체 flan을 생성하고 flan.radiusMatch()로 매칭을 계산한다
# des1에서 des2로의 최대 허용 오차 maxDistance를 적용하여 매칭을 matches에 계산한다. kp1의 각 증짐점에 대해 매칭점의 개수가 다를 수 있고, 없을 수도 있다
# bf = cv2.BFMatcher()
# matches = bf.radiusMatche(des1, des2, maxDistance = 50)
flan = cv2.FlannBasedMatcher_create()
matches = flan.radiusMatch(des1, des2, maxDistance=50) # 200
# print('# of matches = ', len(np.nonzero(np.array(matches, dtype=object))[0]))

#4
# draw_key2image(kp, img) 함수는 특징점 kp를 img영상에 초록색 바운딩 박스와 파란색 원으로 그린다.
# 매칭점 matches의 각 매칭 리스트 radius_match에 대해, len(radius_match) != 0이면(매칭이 존재하면) draw_key2image로 특징점 kp1[radius_match[0].queryIdx]를 src1에 그리고
# radius_match의 각 매칭 m에 대해 draw_key2image()로 특징 kp2[m.trainIdx]를 src2에 그린다.
# cv2.drawMatches()로 radius_match를 dst에 표시하고, dst 영상을 화면에 보여주고 cv2.waitKey()로 멈춘다
def draw_key2image(kp, img):
    x, y = kp.pt
    size = kp.size
    rect = ((x,y), (size, size), kp.angle)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.polylines(img, [box], True, (0, 255, 0), 2)
    cv2.circle(img, (round(x), round(y)), round(size/2), (255, 0, 0), 2)
    return img

for i, radius_match in enumerate(matches):
    if len(radius_match) != 0:
        print('i=', i)
        print('len(matches[{}])={}'.format(i, len(matches[i])))

        src1c = src1.copy()
        draw_key2image(kp1[radius_match[0].queryIdx], src1c)
        src2c = src2.copy()
        for m in radius_match:
            draw_key2image(kp2[m.trainIdx], src2c)
        dst = cv2.drawMatches(src1c, kp1, src2c, kp2, radius_match, None, flags=2)
        cv2.imshow('dst', dst)
        cv2.waitKey()

cv2.waitKey()
cv2.destroyAllWindows()
