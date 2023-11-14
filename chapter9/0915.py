import cv2
import numpy as np

# 1
# SIFT로 특징점 검출과 디스크립터를 계산하고 BFMatcher 또는 FlannBAsedMatcher로 매칭 객체를 생성한다.
# radiusMatch()로 매칭을 계산하고, 투영 변환을 계산한다
src1 = cv2.imread('../data/book1.jpg')
src2 = cv2.imread('../data/book2.jpg')
img1 = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)

# 2
# SIFT 객체 siftF를 생성하고, siftF.detectAndCompute()로 img1, img2의 각가의 특징점 kp1, kp2를 검출하고, 디스크립터 des1, des2를 계산한다.
# len(kp1) = 597, len(kp2) = 881개의 특징점을 검출한다
siftF = cv2.SIFT_create()
kp1, des1 = siftF.detectAndCompute(img1, None)
kp2, des2 = siftF.detectAndCompute(img2, None)
print('len(kp1)={}, len(kp2)={}'.format(len(kp1), len(kp2)))

# 3
# BFMatcher 객체 bf를 생성하고 bf.radiusMatch()로 매칭을 계산하거나, FlannBasedMatcher 객체 flan을 생성하고 flan.radiusMatch()로 매칭을 계산한다
# 허용 오차 임계값 distT = 200으로 초기화하고 des1에서 des2로의 최대 허용 오차 maxDistance = distT를 적용하여 매칭을 matches에 계산한다
# kp1의 각 특징점에 대해 매칭점의 개수가 다를 수 있고 없을 수도 있다
distT = 200 # 500
# bf = cv2.BFMatcher()
# matches = bf.radiusMatch(des1, des2, maxDistance=distT)
flan = cv2.FlannBasedMatcher_create()
matches = flan.radiusMatch(des1, des2, maxDistance=distT)
print('len(matches)=', len(matches))

# 4
good_matches = []
for i, radius_match in enumerate(matches):
    # 4-1
    # 매칭점 matches의 각 매칭 리스트 radius_matches에 대해, len(radius_matches) != 0이면(매칭이 존재하면) 매칭 거리를 기준으로 sort_match에 오름차순 정렬하고,
    # 매칭거리가 가장 작은 sort_matche[0]을 good_matches 리스트에 추가한다. good_matches의 최대값은 len(matches)이다.
    # if len(radius_match) != 0:
    #     sort_match = sorted(radius_match, key=lambda m: m.distance)
    #     good_matches.append(sort_match[0])

    # 4-2
    # 매칭점 matches의 각 매칭 리스트 radius_matches에 대해, len(radius_matches) != 0이면(매칭이 존재하면) radius_match에서 m.distance <100인 매칭으 ㄹgoo_matches 리스트에 추가한다
    if len(radius_match) != 0:
        for m in  radius_match:
            if m.distance < 100: # filter by distance
                good_matches.append(m)

print('len(good_matched)=', len(good_matches))
# dst2 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flag=2)
# cv2.imsho('dst2', dst2)

# 5
src1_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
src2_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# cv2.LMEDS
H, mask = cv2.findHomography(src1_pts, src2_pts, cv2.RANSAC, 3.0)
mask_matches = mask.ravel().tolist()  # list(mask.flatten())

# 6
# 투영 변환을 계산하고, img1의 네 모서리 좌표를 투영 변환시켜 src2에 사변형을 그리고, 매칭 마스크를 사용하여 good_matches에서 인라리어 매칭만을 표시하는 과정이다
h, w = img1.shape
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
pts2 = cv2.perspectiveTransform(pts, H)
src2 = cv2.polylines(src2, [np.int32(pts2)], True, (255, 0, 0), 2)

draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=mask_matches, flags=2)
dst3 = cv2.drawMatches(src1, kp1, src2, kp2, good_matches, None, **draw_params)
cv2.imshow('dst3', dst3)
cv2.waitKey()
cv2.destroyAllWindows()
