import cv2
import numpy as np

#1
# src는 입력 영상이고 마우스로 지정할 마스크 영역을 지정하고, 윤곽선을 검출할 mask 영상, 윤곽선을 이용하여 워터쉐드 분할을 위한 마커 영상 markers를 생성한다.
# src를 dst에 복사한다.
src = cv2.imread('../data/hand.jpg')
# src = cv2.imread('../data/flower.jpg')
mask = np.zeros(shape=src.shape[:2], dtype=np.uint8)
markers = np.zeros(shape=src.shape[:2], dtype=np.int32)
dst = src.copy()
cv2.imshow('dst', dst)

#2
# 마우스 이벤트 핸들러 함수 onMouse()를 정의한다
# param[0]은 mask, param[1]은 dst가 전달된다.
# 마우스 왼쪽버튼을 누르고 움직이면 param[0], param[1]에 반지름 20인 채운 원을 그린다.
def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        if flags & cv2.EVENT_FLAG_LBUTTON:
            cv2.circle(param[0], (x, y), 10, (255, 255, 255), -1)
            cv2.circle(param[1], (x, y), 10, (255, 255, 255), -1)
    cv2.imshow('dst', param[1])
# cv2.setMouseCallback('dst', onMouse, [mask, dst])

#3
# while 반복문에서 키보드 이벤트 처리를 구한현다.
# Esc 키를 누르면 반복문을 종료하고, r키를 누르면 리셋하고, Space Bar 키를 누르면 영역 분할한다.
mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE
while True:
    #3-1
    # cv2.setMouseCallback()로 'dst' 윈도우에 파라미터 [mask, dst]로 onMouse() 이벤트 핸들러를 설정한다.
    # dst가 반복문 안에서 변경되기 때문에 마우스 이벤트 핸들러를 반복문 내에서 설정한다.
    cv2.setMouseCallback('dst', onMouse, [mask, dst])
    key = cv2.waitKey(30)

    if key == 0x1B:
        break;
    #3-2
    # r키를 누르면 리셋하기 위하여 mask의 모든 화소를 0으로 초기화하고, src를 dst에 복사하고 'dst' 윈도우에 표시한다.
    elif key == ord('r'):
        mask[:,:] = 0
        dst = src.copy()
        cv2.imshow('dst', dst)
    #3-3
    # Space Bar 키를 누르면, mask에서 윤곽선을 검출하고, markers를 0으로 초기화하고, cv2.drawContours()로 mask에 윤곽선 contours[i]를 i+1 값으로 채워 넣어, cv2.watershed()의 입력으로 사용한다.
    # c2.watersheD()로 src에서 markers에 표시된 마커 정보를 이용하여 영역을 markers에 분할한다.
    elif key == ord(' '):
        contours, hierarchy = cv2.findContours(mask, mode, method)
        print('len(contours)=', len(contours))
        markers[:,:] = 0
        for i, cnt in enumerate(contours):
            cv2.drawContours(markers, [cnt], 0, i+1, -1)
        cv2.watershed(src, markers)

        #3-4
        # src를 dst에 복사하고, dst[markers == 1] = [0, 0, 255]에 의해 markers에 -1인 경계선을 빨간색 [0,0, 255]로 변경한다.
        # for 문에서, r, g, b에 [0, 255] 사이의 난수를 생성하여 dst[markers == i+1] = [b, g, r]로 markers == i+1인 dst의 화소를 [b, g, r] 컬러로 변경한다.
        # cv2.addWeighted()로 src * 0.4와 dst * 0.6으로 섞어 dst에 저장하고, 'dst' 윈도우에 표시한다.
        dst = src.copy()
        dst[markers == -1] = [0, 0, 255]
        for i in range(len(contours)):
            r = np.random.randint(256)
            g = np.random.randint(256)
            b = np.random.randint(256)
            dst[markers == i+1] = [b, g, r]

        dst = cv2.addWeighted(src, 0.4, dst, 0.6, 0)
        cv2.imshow('dst', dst)

cv2.destroyAllWindows()