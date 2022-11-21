import cv2
import numpy as np

src1 = cv2.imread('../data/lena.jpg', cv2.IMREAD_GRAYSCALE)
src2 = np.zeros(shape=(512,512), dtype=np.uint8)+255

dst1 = 255 - src1 # numpy의 브로드 캐스팅으로 255를 src1 크기의 배열로 확장하고, src1의 각 화소와 뺄셈으로 계산하여 반전 영상 dst1을 생성한다
dst2 = cv2.subtract(src2, src1) # cv2.subtract() 함수를 사용하여 (src2-src1) 연산으로 src1의 화소값을 반전하여 반전 영상 dst2를 계산한다
dst3 = cv2.compare(dst1, dst2, cv2.CMP_NE) # cv2.compare() 함수로 dst1과 dst2의 각 화소를 cv2.CMP_NE(not equal to) 비교하여, 참이면 255, 거짓이면 0을 dst 영상의 각 화소에 출력한다. dst1과 dst2는 모든 화소에서 같은 값을 갖는다
n = cv2.countNonZero(dst3) # dst3에서 0이 아닌 화소를 카운트하여 반환한다
print('n = ', n) # dst3의 화소는 모두 0이기 때문에, n = 0의 결과를 확인할 수 있다

cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()