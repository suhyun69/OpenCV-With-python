import cv2
import numpy as np

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            cv2.rectangle(param[0], (x-5, y-5), (x+5, y+5), (255, 0, 0))
        else:
            cv2.circle(param[0], (x, y), 5, (255, 0, 0), 3)
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(param[0], (x, y), 5, (0, 0, 255), 3)
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        param[0] = np.zeros(param[0].shape, np.uint8) + 255
    cv2.imshow("img", param[0])

img = np.zeros(shape=(512, 512, 3), dtype=np.uint8) + 255
cv2.imshow('img', img)
cv2.setMouseCallback('img', onMouse, [img]) # cv2.setMouseCallBack() 함수로 'img' 윈도우의 마우스 이벤트 핸들러로 onMouse() 함수를 설정하고, 매개변수 param에 img 영상의 리스트 [img]를 전달한다
cv2.waitKey()
cv2.destroyAllWindows()