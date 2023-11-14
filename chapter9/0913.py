import cv2
import numpy as np

# 1
# SIFT로 특징점 검출과 디스크립터를 계산하고, BFMatcher 또는 FlannBasedMatcher로 매칭객체를 생성한다
# knnMatcher로 매칭을 계산한다
src1 = cv2.imread('../data/book1.jpg')
src2 = cv2.imread('../data/book2.jpg')
img1 = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)

# 2
# SIFT 객체 siftF를 생성하고, siftdetectAndCompute()로 img1, img2의 각각의 특징점 kp1, kp2를 검출하고, 디스크립터 des1, des2를 계산한다
# img1, kp1, des1은 질의 영상, 특징점, 디스크립터이고
# img2, kp2, des2은 학습 영상, 특징점, 디스크립터로 사용한다.
# 특징 매칭은 질의 영상의 각 특징점에 대해 매칭하는 학습 영상의 특징점을 디스크립터로 사용하여 찾는 것이 목적이다
# len(kp1) = 597, len(kp2) = 881개의 특징점을 검출한다
siftF = cv2.SIFT_create()
kp1, des1 = siftF.detectAndCompute(img1, None)
kp2, des2 = siftF.detectAndCompute(img2, None)

# 3-1
# cv2.BFMatcher()로 매칭 객체 bf를 생성한다. bf.knnMatch(des1, des2, k=2)로 des1에서 des2로의 k=2개의 매칭 matches를 계산한다
# SIFT는 len(matches) = 597개의 매칭을 검출한다
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# 3-2
# FlannBasedMatcher 객체 flan을 생성하고, flan.knnMatcher(des1, des2, k=2)로 des1에서 des2로의 k=2개의 매칭 matches를 계산한다
# SIFT는 len(matches) = 597개의 매칭을 검출한다
# flan = cv2.FlannBasedMatcher_create()
# matches = flan.knnMatch(dex1, des2, k=2)

# 3-3
# knnMatch()에서 k=2 매칭은 각 특징점에 대해 가장 가까운 이웃 2개를 매칭점으로 검출한다
# 예를 들어 kp1[0]에 대한 매칭은 matches[0][0], matches[0][1]의 2개이다
# 각각의 매칭에 대해 matches[0][0].distance <= matches[0][1].distance, ..., matches[i][0].distance <= matches[i][1].distance이다.
# cv2.drawMatchesKnn()으로 matches를 dst에 그린다.
print('len(matches)=', len(matches))
for i, m in enumerate(matches[:3]):
    for j, n  in enumerate(m):
        print('matches[{}][{}]=(queryIdx:{}, trainIdx:{}, distance:{})'.format(i, j, n.queryIdx, n.trainIdx, n.distance))
dst = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=0)
# cv2.imshow('dst', dst)

# 4: find good matches
# NNDR(Nearest neighbor distance ratio)을 ㅏㅅ용하여 matches에서 좋은 매칭 good_matches을 찾는다. d1은 가장 가까운 이웃가지의 거리이고, d2는 두 번째로 가까운 이웃까지의 거리이다
# NNDR = d1/d2이 작으면 좋은 매칭으로 판단한다
# matches에서 nndrRatio < 0.45이면 good_matches에 첫 번째 매칭 m을 저장한다.
# nndrRatio.가 작을수록 좋은 매칭의 개수가 적게 검출된다
nndrRatio = 0.45
good_matches = [f1 for f1, f2 in matches if f1.distance < nndrRatio * f2.distance]

# good_matches = []
# for f1, f2, in matches: # k = 2
#     if f1.distance < nndrRatio * f2.distance:
#         good_matches.append(f1)

print('len(good_matches)=', len(good_matches))
if len(good_matches) < 5:
    print('sorry, too small good matches')
    exit()

#5
# 투영 변환을 계산하고, img1의 네 모서리 좌표를 투영 변환시켜 src2에 사변형을 그리고, 매칭 마스크를 사용하여 good_matches에서 인라리어 매칭만을 표시하는 과정이다
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
dst2 = cv2.drawMatches(src1, kp1, src2, kp2, good_matches, None, **draw_params)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()