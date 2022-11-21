import cv2
src = cv2.imread('../data/lena.jpg')

b, g, r = cv2.split(src)
dst = cv2.merge([b, g, r]) # 리스트 [b, g, r]을 dst에 채널 합성한다. 리스트의 항목 순서 b, g, r의 순서는 채널 순서로 중요하다

print(type(dst))
print(dst.shape)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()