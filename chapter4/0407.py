import cv2

src = cv2.imread('../data/lena.jpg', cv2.IMREAD_GRAYSCALE)
roi = cv2.selectROI(src)
print('roi = ', roi)

# roi = cv2.selectROI(src)는 디폺트 'ROI selector' 윈도우에 src 영상을 표시하고,
# 마우스 클릭과 드래그로 ROI를 선택하고, 스페이스/엔터키를 누르면 선택 영역을 roi에 반환한다
# 만약 마우스로 ROI를 선택하지 않고 스페이스바/엔터키를 누르면 roi = (0, 0, 0, 0)을 반환한다

if roi != (0, 0, 0, 0):
    img = src[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    cv2.imshow('Img', img)
    cv2.waitKey()

cv2.destroyAllWindows()