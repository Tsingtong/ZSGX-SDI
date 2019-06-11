# coding:utf-8
import cv2
import time
import numpy as np
from utils import calculateALL, status

if __name__ == '__main__':
    time1 = 0.1
    time2 = 0.1

    # 获取摄像头对象
    cap1 = cv2.VideoCapture('in.mpg')
    cap2 = cv2.VideoCapture('out.mpg')

    # 设置窗口对象
    win_name = "output"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    fps1 = round(cap1.get(cv2.CAP_PROP_FPS))
    print(fps1 / 60)

    while cap1.isOpened():
        while cap2.isOpened():
            break
        else:
            time.sleep(1)
        break

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        frame1 = cv2.resize(frame1, (768, 432), interpolation=cv2.INTER_CUBIC)
        frame2 = cv2.resize(frame2, (768, 432), interpolation=cv2.INTER_CUBIC)

        time1 = time.time()
        frame = np.vstack((frame1, frame2))
        MSE, SSIM = calculateALL(frame1, frame2)
        seconds = time1 - time2
        result = status(MSE, SSIM)
        print(result)
        time2 = time1
        cv2.imshow(win_name, frame)  # 显示摄像头当前帧内容
        # Calculate frames per second
        fps = 1 / seconds
        print('Current FPS:', round(fps))
        if cv2.waitKey(1) & 0xFF == ord('p'):  # ord() 将ASCLL码值转换为字符
            while True:
                time.sleep(0.5)
                if cv2.waitKey(1) & 0xFF == ord('c'):  # ord() 将ASCLL码值转换为字符
                    break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
