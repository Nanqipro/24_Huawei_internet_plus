# import os
# import copy
# import queue
# import traceback
# import numpy as np
# import time
# import gc
# import threading
# import sys
# import psutil
# import argparse
# import cv2
#
# from model1.onnx_detect import ONNX_engine
# import our_tool
#
#
# class fatigue_driving_detection():  # 本地
#     def __init__(self, model_name, model_path):
#         self.onnx_model = ONNX_engine(weights_path='v4_c7_320.onnx', size=320)  # 使用onnx模型
#         self.model_name = model_name
#         self.model_path = model_path
#
#         self.width = 1920
#         self.height = 1080
#         self.fps = 30
#         self.frameNum_3s = self.fps * 3  # 3s有几帧
#         self.skip_step = 6  # 跳帧间隔
#
#         # 转头、闭眼、张嘴、打手机的连续帧计数值
#         self.look_around_frame = 0
#         self.eyes_closed_frame = 0
#         self.mouth_open_frame = 0
#         self.use_phone_frame = 0
#
#         self.model1_face = False  # 是否检测到人脸
#         self.device = 'cpu'  # 使用CPU推理
#         self.failures = 0
#
#     def reset_CountValue_between_frames(self):
#         self.is_normal = False  # 5种行为
#         self.is_closeEye = False
#         self.is_openMouth = False
#         self.is_usePhone = False
#         self.is_turned = False
#
#         self.model1_openEye = 0
#         self.model1_closeEye = 0
#         self.model1_eyeNum = 0
#
#         self.model1_mouthNum = 0
#         self.model1_openMouth = False
#         self.model1_closeMouth = False
#         self.model1_usePhone = False
#         self.model1_turned = False
#         self.model1_turnedDown = False
#
#     # 模型1，图像检测，检测出现的行为
#     def model_1_detect(self, frame):
#         bbox = self.onnx_model.preprocess(frame)
#         if len(bbox) > 0:
#             for box in bbox:
#                 if box[0] == 0:  # 人脸
#                     self.model1_face = True
#                 elif box[0] == 1:  # 手机
#                     self.model1_usePhone = True
#                 elif box[0] == 2:  # 睁眼
#                     self.model1_openEye += 1
#                     self.model1_eyeNum += 1
#                 elif box[0] == 3:  # 闭眼
#                     self.model1_closeEye += 1
#                     self.model1_eyeNum += 1
#                 elif box[0] == 4:  # 打哈欠
#                     self.model1_openMouth = True
#                     self.model1_mouthNum += 1
#                 elif box[0] == 5:  # 闭嘴
#                     self.model1_closeMouth = True
#                     self.model1_mouthNum += 1
#                 elif box[0] == 6:  # 转头
#                     self.model1_face = True
#                     self.model1_turned = True
#
#     def judge_action(self):
#         if self.model1_closeEye > 0:
#             self.is_closeEye = True
#         else:
#             self.is_closeEye = False
#
#         if self.model1_closeMouth or self.model1_mouthNum == 0:
#             self.is_openMouth = False
#         else:
#             self.is_openMouth = True
#
#         if self.model1_usePhone:
#             self.is_usePhone = True
#
#         if self.model1_turned:
#             self.is_turned = True
#
#     def process_frame(self, frame):
#         try:
#             self.reset_CountValue_between_frames()
#             # # 画面剪裁，只处理主驾驶位
#             # frame = frame[:, int(self.width * 0.33): int(self.width), :]
#             self.model_1_detect(frame)  # 模型检测
#             self.judge_action()  # 判断行为
#
#             # 根据检测结果，在frame上显示相关信息
#             if self.is_closeEye:
#                 cv2.putText(frame, 'Fatigue Detected: Closed Eyes', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
#                             2)
#             if self.is_openMouth:
#                 cv2.putText(frame, 'Fatigue Detected: Yawning', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#             if self.is_usePhone:
#                 cv2.putText(frame, 'Fatigue Detected: Using Phone', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
#                             2)
#             if self.is_turned:
#                 cv2.putText(frame, 'Fatigue Detected: Looking Away', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
#                             (0, 0, 255), 2)
#
#         except Exception as e:
#             print(f"Error processing frame: {str(e)}")
#             traceback.print_exc()
#
#         return frame
#
#
# # 获取摄像头实时检测疲劳驾驶
# def run_camera_detection():
#     detection = fatigue_driving_detection("model name", "path_to_model")
#
#     # 获取系统默认摄像头
#     cap = cv2.VideoCapture(0)
#
#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         return
#
#     # 无限循环，处理每一帧画面
#     while cap.isOpened():
#         # 获取画面
#         success, frame = cap.read()
#         if not success:
#             print('Error: Unable to read from camera.')
#             break
#
#         # 处理帧，调用疲劳检测
#         frame = detection.process_frame(frame)
#
#         # 显示处理后的画面
#         cv2.imshow('Fatigue Detection', frame)
#
#         # 按'q'或'ESC'键退出
#         if cv2.waitKey(1) in [ord('q'), 27]:
#             break
#
#     # 释放摄像头和关闭窗口
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     run_camera_detection()


import os
import copy
import queue
import traceback
import numpy as np
import time
import gc
import threading
import sys
import psutil
import cv2

from model1.onnx_detect import ONNX_engine
import our_tool


class fatigue_driving_detection():  # 本地
    def __init__(self, model_name, model_path):
        self.onnx_model = ONNX_engine(weights_path='v4_c7_320.onnx', size=320)  # 使用onnx模型
        self.model_name = model_name
        self.model_path = model_path

        self.width = 1920
        self.height = 1080
        self.fps = 30
        self.frameNum_3s = self.fps * 3  # 3s有几帧
        self.skip_step = 6  # 跳帧间隔

        # 转头、闭眼、张嘴、打手机的连续帧计数值
        self.look_around_frame = 0
        self.eyes_closed_frame = 0
        self.mouth_open_frame = 0
        self.use_phone_frame = 0

        self.model1_face = False  # 是否检测到人脸
        self.device = 'cpu'  # 使用CPU推理
        self.failures = 0

    def reset_CountValue_between_frames(self):
        self.is_normal = False  # 5种行为
        self.is_closeEye = False
        self.is_openMouth = False
        self.is_usePhone = False
        self.is_turned = False

        self.model1_openEye = 0
        self.model1_closeEye = 0
        self.model1_eyeNum = 0

        self.model1_mouthNum = 0
        self.model1_openMouth = False
        self.model1_closeMouth = False
        self.model1_usePhone = False
        self.model1_turned = False
        self.model1_turnedDown = False

    # 模型1，图像检测，检测出现的行为
    def model_1_detect(self, frame):
        bbox = self.onnx_model.preprocess(frame)
        if len(bbox) > 0:
            for box in bbox:
                if box[0] == 0:  # 人脸
                    self.model1_face = True
                elif box[0] == 1:  # 手机
                    self.model1_usePhone = True
                elif box[0] == 2:  # 睁眼
                    self.model1_openEye += 1
                    self.model1_eyeNum += 1
                elif box[0] == 3:  # 闭眼
                    self.model1_closeEye += 1
                    self.model1_eyeNum += 1
                elif box[0] == 4:  # 打哈欠
                    self.model1_openMouth = True
                    self.model1_mouthNum += 1
                elif box[0] == 5:  # 闭嘴
                    self.model1_closeMouth = True
                    self.model1_mouthNum += 1
                elif box[0] == 6:  # 转头
                    self.model1_face = True
                    self.model1_turned = True

    def judge_action(self):
        if self.model1_closeEye > 0:
            self.is_closeEye = True
        else:
            self.is_closeEye = False

        if self.model1_closeMouth or self.model1_mouthNum == 0:
            self.is_openMouth = False
        else:
            self.is_openMouth = True

        if self.model1_usePhone:
            self.is_usePhone = True

        if self.model1_turned:
            self.is_turned = True

    def process_frame(self, frame):
        try:
            self.reset_CountValue_between_frames()
            # # 画面剪裁，只处理主驾驶位
            # frame = frame[:, int(self.width * 0.33): int(self.width), :]
            self.model_1_detect(frame)  # 模型检测
            self.judge_action()  # 判断行为

            # 根据检测结果，在frame上显示相关信息
            if self.is_closeEye:
                cv2.putText(frame, 'Fatigue Detected: Closed Eyes', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                            2)
            if self.is_openMouth:
                cv2.putText(frame, 'Fatigue Detected: Yawning', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if self.is_usePhone:
                cv2.putText(frame, 'Fatigue Detected: Using Phone', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                            2)
            if self.is_turned:
                cv2.putText(frame, 'Fatigue Detected: Looking Away', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)

        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            traceback.print_exc()

        return frame


# 获取摄像头实时检测疲劳驾驶
def run_camera_detection():
    detection = fatigue_driving_detection("model name", "path_to_model")

    # 获取系统默认摄像头
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # 无限循环，处理每一帧画面
    while cap.isOpened():
        # 获取画面
        success, frame = cap.read()
        if not success:
            print('Error: Unable to read from camera.')
            break

        # 处理帧，调用疲劳检测
        frame = detection.process_frame(frame)

        # 显示处理后的画面
        cv2.imshow('Fatigue Detection', frame)

        # 按'q'或'ESC'键退出
        if cv2.waitKey(1) in [ord('q'), 27]:
            break

    # 释放摄像头和关闭窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera_detection()
