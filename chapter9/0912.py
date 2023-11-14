import cv2
import numpy as np

# 1
# ORB 또는 BRISK로 특징점 검출과 디스크립터를 계산하고, BFMatcher 또는 FlannBasedMatcher로 매칭객체를 생성한다
# DescriptorMatcher.match() 메서드로 매칭을 계산한다
src1 = cv2.imread('../data/book1.jpg')
src2 = cv2.imread('../data/book2.jpg')
img1 = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)

# 2-1
# ORB 객체 orbF를 생성하고, orbf.detectAndCompute()로 img1, img2의 각가의 특징점 kp1, kp2를 검출하고, 디스크립터 des1, des2를 계산한다
# img1, kp1, des1은 질의 영상, 특징점, 디스크립터이고
# img2, kp2, des2은 학습 영상, 특징점, 디스크립터로 사용한다.
# 특징 매칭은 질의 영상의 각 특징점에 대해 매칭하는 학습 영상의 특징점을 디스크립터로 사용하여 찾는다
orbf = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orbf.detectAndCompute(img1, None)
kp2, des2 = orbf.detectAndCompute(img2, None)

# 2-2
# BRISK 객체 briskF를 생성하고, briskF.detectAndCompute()로 img1, img2의 각가의 특징점 kp1, kp2를 검출하고, 디스크립터 des1, des2를 계산한다
# briskF = cv2.BRISK_create()
# kp1, des1 = briskF.detectAndCompute(img1, None)
# kp2, des2 = briskF.detectAndCompute(img2, None)

# 3-1
# BFMatcher_create()에서 cv2.NORM_HAMMING로 이진 디스크립터의 비트가 일치하지 않는 개수를 거리로 사용하고, crossCheck = True로 des1에서 des2로 매칭을 찾고,
# des2의 매칭에서 des1으로 매칭을 확인하는 매칭 객체 bf를 생성한다
# bf.match()로 dex1에서 dex2로의 매칭 matches를 계산한다. 
# bf.match()는 매칭을 하나씩만 검출하기 때문에 matches는 각 항목이 DMatch 객체인 리스트이다
bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# 3-2
# FlannBasedMatcher 객체 flan을 생성하고, flan.match()로 des1에서 des2로의 aocld matches를 계산한다
# FlannBasedMatchr는 32비트 실수 디스크립터를 입력받기 때문에 np.flot32(des1), np.float32(des2)로 변경하여 flan.match()로 cv2.NORM_L2 거리에 의해 매칭한다
# flan = cv2.FlannBasedMatcher_create()
# matches = flan.match(np.float32(des1), np.float32(des2))

# 3-3
# matches를 매칭 거리가 작은 값이 먼저 나온느 오름차순으로 정렬한다.
# 예를 들어 matches[0] = <DMatch 0704D3C8>이고, matches[0].queryIdx = 310, matches[0].trainIdx = 215, matches[0].distance = 10.0, matches[0].imgIdx = 0이다.
# 이것은 특징점 kp1[314]와 kp2[215]가 거리 matches[0].distance = 10.0로 매칭하는 것을 의미한다.
# matches[0].imgIdx = 0으 ㄴ학습 영상의 인덱스이다. 학습영상이 img2 하나만 사용하므로 0이다.
# 학습영상을 영상의 리스트로 사용할 수 있다. for문으로 3개의 matches 정보를 출력한다. 
# 거리는 작을수록 매칭이 잘된 매칭디ㅏ. matches의 거리가 작은 값이 먼저 나오는 오름차순으로 정렬된 것을 확인할 수 있다

# 4
# 그러므로 matches[0].distance가 minDist이다. filter()를 사용하여 minDist의 5배(임의로 설정)보다 작은 거리의 매칭을 필터링하여 good_matches에 저장한다
# len(good_matches) < 5이면 프로그램을 종료한다.
# 투영 변환을 위해서는 최소 4점이면 되지만, 정확시 4점일 경우 대부분의 경우 투영 변환이 틀어진다. 매칭점이 많을수록 투여 변환 계산이 보다 정확하다
# cv2.drawMatches()로 (img1, kp1)에서 (img2, ip2)로의 매칭 good_matches를 dst에 표시한다
# flags = 2이므로 매칭이 없는 특징점은 원으로 표시하지 않는다.
matches = sorted(matches, key=lambda m: m.distance)
print('len(matches)=', len(matches))
for i, m in enumerate(matches[:3]):
    print('matches[{}]=(queryIdx:{}, trainIdx:{}, distance:{}'.format(i, m.queryIdx, m.trainIdx, m.distance))

minDist = matches[0].distance
good_matches = list(filter(lambda m: m.distance < 5 * minDist, matches))
print('len(good_matches)=', len(good_matches))
if len(good_matches) < 5:
    print('sorry, too small good matches')
    exit()

dst = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)
cv2.imshow('dst', dst)

# 5
# good_matche의 질의 인덱스(queryIdx)와 kp1을 이용하여 img1의 특징점 좌표를 src1_pts 배열에 저장하고
# 학습 인덱스스(.trainIdx)와 kp2를 이용하여 img2의 특징점 좌표를 src2_pts 배열에 저장한다
# cv2.findHomography()로 src1_pts에서 src2_pts로의 투영 변환 H을 cv2.RANSAC 방법으로 최대허용오차 ransacReprojThreshold = 3을 적용하여 계산한다
# mask에 1에 대응하는 매칭점은 인라이어(inlier)이고, 0에 대응하는 매칭점은 아웃라이어(outlier)이다.
# masrk.ravel().tolist()로 mask를 1차뤈 리스트 mask_matches로 변환한다
src1_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
src2_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# cv2.LMEDS
H, mask = cv2.findHomography(src1_pts, src2_pts, cv2.RANSAC, 3.0)
mask_matches = mask.ravel().tolist()  # list(mask.flatten())

# 6
# img1의 가로(w). 세로(h) 크기를 이용하여 영상의 모서리 좌표 [0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]을 (4, 1, 2)의 배열 pts에 저장한다
# cv2.perspectiveTransform()로 pts에 투영 변환 H을 적용하여 pts2로 변환한다. 즉, 매칭점 사이의 투영 변환 H을 계산하여, img1의 네 모서리 좌표를 투영 변환하여 src2(또는 img2)의 좌표 pts2로 변환한다
# cv2.polylines()로 pts2를 파란색 사변형으로 src2에 표시한다
# draw_params 사전에 matchColor을 초록색으로, singlePointColor = None로 매칭되지 않는 점의 색은 지정하지 않고
# 매칭 마스크를 mask_matches로 지정하여, 투영 변환에서 good_matches의 인라이어 매칭점만 cv2.drawMatches()로 dst2에 표시한다
h, w = img1.shape
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
pts2 = cv2.perspectiveTransform(pts, H)
src2 = cv2.polylines(src2, [np.int32(pts2)], True, (255, 0, 0), 2)

draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=mask_matches, flags=2)
dst2 = cv2.drawMatches(src1, kp1, src2, kp2, good_matches, None, **draw_params)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()

