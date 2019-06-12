# coding:utf-8
import cv2
import time
import numpy as np
from utils import calculateALL, status


def toTime(videotime):
    m, s = divmod(videotime, 60)
    h, m = divmod(m, 60)
    a = "%02d-%02d-%02d" % (h, m, s)
    return a


if __name__ == '__main__':
    time1 = 0.1
    time2 = 0.1

    # 获取摄像头对象
    cap1 = cv2.VideoCapture('cctv_1080p.mp4')
    cap2 = cv2.VideoCapture('cctv_720p.mp4')

    # 设置窗口对象
    win_name = "output"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # 设置错误图片路径
    image_path = './failure/'
    frame_id = 0

    fps1 = round(cap1.get(cv2.CAP_PROP_FPS))
    # print(fps1 / 60)

    while cap1.isOpened():
        while cap2.isOpened():
            break
        else:
            time.sleep(1)
        break

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        frame1 = cv2.resize(frame1, (1280, 720), interpolation=cv2.INTER_CUBIC)

        time1 = time.time()
        frame = np.vstack((frame1, frame2))
        MSE, SSIM = calculateALL(frame1, frame2)
        seconds = time1 - time2
        videotime = frame_id // 25
        result = status(MSE, SSIM)
        videotime = toTime(videotime)
        print('video time:', videotime)

        if result == "正常":
            print('frame_id:'+str(frame_id)+' '+result)
        elif result == "警告":
            cv2.imwrite(image_path+str(frame_id)+'-['+videotime+']-[warning]'+'-SSIM['+str(SSIM)+']-MSE['+str(MSE)+'].jpg', frame)
            print('frame_id:'+str(frame_id)+' '+result)
        elif result == "错误":
            cv2.imwrite(image_path+str(frame_id)+'-['+videotime+']-[error]'+'-SSIM['+str(SSIM)+']-MSE['+str(MSE)+'].jpg', frame)
            print('frame_id:'+str(frame_id)+' '+result)

        time2 = time1
        cv2.imshow(win_name, frame)  # 显示摄像头当前帧内容
        # Calculate frames per second
        fps = 1 / seconds
        # print('Current FPS:', round(fps))
        if cv2.waitKey(1) & 0xFF == ord('p'):  # ord() 将ASCLL码值转换为字符
            while True:
                time.sleep(0.5)
                if cv2.waitKey(1) & 0xFF == ord('c'):  # ord() 将ASCLL码值转换为字符
                    break
        frame_id += 1

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
