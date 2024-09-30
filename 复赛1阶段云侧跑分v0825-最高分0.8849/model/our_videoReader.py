import time
import traceback
import cv2

class VideoReader():
    def __init__(self, capture) -> None:
        self.capture = capture      # video path
        self.cap = None
        self.ret = False
        try:
            self.cap = cv2.VideoCapture(self.capture)           # 创建VideoCapture实例
            if self.cap.isOpened():             # 成功
                self.ret = True
            else:                               # 失败
                reinit_try_count = 0
                need_reinit_times = 10          # 重试10次
                while not self.cap.isOpened() and reinit_try_count < need_reinit_times:
                    self.cap.release()          # 先释放
                    time.sleep(0.1)
                    self.cap.open(self.capture) # 再重新打开
                    reinit_try_count += 1

                if self.cap.isOpened():         # 当重试成功
                    self.ret = True
                else:                           # 当重试失败，释放release
                    self.cap.release()
                    print('reinit try times:', reinit_try_count,"; init AND reinit cap failed! Exit!")
        except Exception as e:
            traceback.print_exc()
            print('jhj [Exception] failed to init_VideoCapture to open video')

    
    # 获取视频信息：height、width、fps、total frame
    def get_capInfo(self):
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.video_totalFrameNum = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print('Get height:', self.height, 'width:', self.width, 'fps:', self.fps, 'video_frame_count:', self.video_totalFrameNum, "\n")
        return self.height, self.width, self.fps, self.video_totalFrameNum
    
    # set pos 并且 read
    def set_and_read(self, frame_num):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num -1)
        ret, frame = self.cap.read()
        return ret, frame
    
    def grab_forword_read(self, step: int):  # 确保step>0
        if step > 150:                  # step小于150,200时,grab比set pos快; set 表示是否在step大于150的时候调用set来处理
            frame_idx = self.cap.get(cv2.CAP_PROP_POS_FRAMES) + step
            ret, frame = self.set_and_read(frame_idx)
        elif step > 1:
            for i in range(step-1):     # 0-step-2,共执行step-1次
                self.cap.grab()
            ret, frame = self.cap.read()
        elif step == 1:
            ret, frame = self.cap.read()
        else:
            print('step connot <= 0')
            ret = False
            frame = None
        return ret, frame
    
    # 释放cap对象
    def release(self):
        self.cap.release()
    
    # 帧率
    def get_fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)
    
    # 高
    def get_height(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    # 宽
    def get_width(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    
    # 视频帧数
    def get_frameCount(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    def get_nowPos(self):
        return self.cap.get(cv2.CAP_PROP_POS_FRAMES)