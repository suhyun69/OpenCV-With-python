import cv2
import numpy as np

#1
# 원, 삼각형, 직사각형 물체가 있는 'refShapes.jpg' 참조 영상을 ref_src에 읽고, 이진 영상을 ref_bin에 생성하고 윤곽선을 ref_contours에 검출한다.
# testShape1.jpg 테스트 영상을 test_src에 읽고, 이진 영상을 test_bin에 생성하고 윤곽선을 test_contours에 검출한다
ref_src = cv2.imread('../data/refShapes.jpg')
ref_gray = cv2.cvtColor(ref_src, cv2.COLOR_BGR2GRAY)
ret, ref_bin = cv2.threshold(ref_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

test_src = cv2.imread('../data/testShapes1.jpg')
test_gray = cv2.cvtColor(test_src, cv2.COLOR_BGR2GRAY)
ret, test_bin = cv2.threshold(test_gray, 0, 255,  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE
ref_contours, _ = cv2.findContours(ref_bin, mode, method)
test_contours, _ = cv2.findContours(test_bin, mode, method)

#2
# 3가지 기준 모양을 구별하여 표시하기 위해 colors에 컬러를 생성한다. ref_src를 복사한 ref_dst에 참조 영상의 윤곽선 ref_contours를 colorf 컬러로 표시한다
ref_dst = ref_src.copy()
colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
for i, cnt in enumerate(ref_contours):
    cv2.drawContours(ref_dst, [cnt], 0, colors[i], 2)

#3: shape matching
# test_contours의 각 윤곽선 cnt1에 대하여 cv2.matchShapes()로 ref_contours의 각 윤곽선 cnt2의 모양 매칭 결과를 matches 배열예 계산하고, 최소값 첨자 k를 찾는다
# test_contours의 각 윤곽선 cnt1을 colors[k] 색상으로 test_src에 복사한 test_dst에 표시한다
# 
test_dst = test_src.copy()
method = cv2.CONTOURS_MATCH_I1
for i, cnt1 in enumerate(test_contours):
    matches = []
    for cnt2 in ref_contours:
        ret = cv2.matchShapes(cnt1, cnt2, method, 0)
        matches.append(ret)
    k = np.argmin(matches)
    cv2.drawContours(test_dst, [cnt1], 0, colors[k], 2)

cv2.imshow('ref_dst', ref_dst)
cv2.imshow('test_dst', test_dst)

cv2.waitKey()
cv2.destroyAllWindows()