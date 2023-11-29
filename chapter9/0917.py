import cv2
from matplotlib import pyplot as plt

#1
# cv2.HOGDescriptor()로 기본값의 hog 객체를 생성하고, hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())로
# 사람 검출을 위해 미리 학습된 계수로 SVM 분류기를 설정한다
src = cv2.imread('../data/people.png')
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#2
# hog.detect(src)로 src 영상에서 사람을 검출하면 len(loc1) = 0으로 검출할 수 없다
# hog.detect()는 HOG 디스크립터를 계산할 때의 크기와 정확히 같아야 검출할 수 있다
loc1, weights1 = hog.detect(src)
print('len(loc1)=', len(loc1))
dst1 = src.copy()
w, h = hog.winSize
for pt in loc1:
    x, y = pt
    cv2.rectangle(dst1, (x, y), (x+w, y+h), (255, 0, 0), 2)
cv2.imshow('dst1', dst1)

#3
# hog.detectMultiScale(src)로 src 영상에서 다중 스케일로 사람을 검출하면 len(loc2) = 3개의 영역이 검출되고
# 검출된 영역을 사각형으로 표시한다
dst2 = src.copy()
loc2, weights2 = hog.detectMultiScale(src)
print('len(loc2)=', len(loc2))
for rect in loc2:
    x, y, w, h = rect
    cv2.rectangle(dst2, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.imshow('dst2', dst2)

#4
# hog.detectMultiScale(src, winStride=(1, 1), padding=(8, 8))로 src 영상에서 다중 스케일로 사람을 검출하면
# len(loc3) = 6개의 영역이 검출되고, 검출된 영역을 사각형으로 표시한다
# 테두리 영역 패딩으로 검출되지 않은 사람을 검출할 수 있다
# 사람이 아닌 파란색 사각형은 신뢰도 0.37325611로 사람이 아님에도 검출되었다
dst3 = src.copy()
loc3, weights3 = hog.detectMultiScale(src, winStride=(1, 1), padding=(8, 8))
print('len(loc3)=', len(loc3))
print('weights3=', weights3)
for i, rect in enumerate(loc3):
    x, y, w, h = rect
    if weights3[i] > 0.5:
        cv2.rectangle(dst3, (x, y), (x+w, y+h), (0, 0, 255), 2)
    else:
        cv2.rectangle(dst3, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('dst3', dst3)
cv2.waitKey()
cv2.destroyAllWindows()