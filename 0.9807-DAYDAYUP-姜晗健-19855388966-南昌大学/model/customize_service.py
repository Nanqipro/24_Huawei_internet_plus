import os
import queue
import traceback
import time
import gc
import threading
import psutil

from model1.onnx_detect import ONNX_engine
import our_tool
from our_videoReader import VideoReader

# from model_service.pytorch_model_service import PTServingBaseService # 提交
# class fatigue_driving_detection(PTServingBaseService): # 提交
class fatigue_driving_detection():  # 本地
    def __init__(self, model_name, model_path):
        self.onnx_model = ONNX_engine(weights_path='/home/mind/model/v4_c7_320.onnx', size=320) # 提交！！！！看清楚权重
        self.model_name = model_name
        self.model_path = model_path
        self.capture = '/home/mind/model/test.mp4'

        self.width = 1920
        self.height = 1080
        self.fps = 30
        self.frame_3s = self.fps * 3        # 3s有几帧
        self.skip_step = 6     # 跳帧间隔

        # 转头、闭眼、张嘴、打手机的连续帧计数值
        self.look_around_frame = 0
        self.eyes_closed_frame = 0
        self.mouth_open_frame = 0
        self.use_phone_frame = 0

        self.model1_face = False   # 是否检测到人脸
        # self.model1_imgsz = 640
        self.device = 'cpu'  # 大赛后台使用CPU判分
        self.failures = 0

        self.video_count = 0  # 一次批量服务的视频计数
        self.file_name = None

    def _preprocess(self, data):
        # data: {'input_video': OrderedDict([('day_man_101_41_6.mp4', <_io.BytesIO object at 0x7fd95e91c770>)])}
        self.video_count += 1
        for k, v in data.items():
            for file_name, bytesIO_object in v.items():
                print('file_name:', file_name, 'jhj video count:', self.video_count)
                self.file_name = file_name
                try:
                    with open(self.capture, 'wb') as f:
                        file_content_bytes = bytesIO_object.read()
                        f.write(file_content_bytes)
                except Exception:
                    traceback.print_exc()
                    return {"message": "There was an error read or write file"}
        return 'ok'

    # 推理部分
    def _inference(self, data):
        print(data)
        self.result = {"result": {"category": 0, "duration": 6000}}
        
        # 初始化，重置不同视频间的计数值，便于批量测试
        self.reset_CountValue_between_videos() 
        
        # 初始化，视频capture
        self.reader = VideoReader(self.capture)
        if self.reader.ret == False:
            return self.result            
        
        # 初始化，获取视频信息
        self.height, self.width, self.fps, self.video_frame_count = self.reader.get_capInfo() # 获取视频信息
        self.frame_3s = self.fps * 3    # 3帧长度
        # self.skip_step = 6

        # 初始化，设置跳帧间隔，每秒取几帧处理，0604
        deal_frame_ps = 1
        self.skip_step = int(self.fps/float(deal_frame_ps)) # 按照每秒取多少帧来处理，来确定step
        self.frame_3s = int(deal_frame_ps * self.skip_step * 3)
        # self.frame_3s = int(self.fps * 3 / deal_frame_ps) * deal_frame_ps

        # 初始化，帧队列
        self.frame_queue = queue.Queue(maxsize=3)
        
        # 初始化，创建一个子线程用于取帧，异步IO
        io_thread = threading.Thread(target=self.read_frame_worker, args=(self.skip_step, ))
        io_thread.daemon = True  # 守护线程, 随进程结束而结束
        io_thread.start()

        # 初始化，第一个视频帧，从帧队列中获取一帧
        frame = self.frame_queue.get()
        
        frame_idx = 0       # 开始的时候index为0
        frame_idx += self.skip_step

        # 模型推理部分
        # 开始记录推理时间，参考baseline的计时位置，在第一张图片进行推理之前
        infer_start_time = time.time()

        while frame is not None:
            self.reset_CountValue_between_frames() # 重置帧间的计数值
            print(f'start dealing frame_idx = {frame_idx}')
            try:
                frame = frame[:, int(self.width /3): int(self.width), :] # 画面剪裁，主驾驶位； [h, w, 通道]

                self.model_1_detect(frame)  # 模型1，图像检测

                self.calculates_timeCount_of_class_v9() # 0602 ，1111，在v7的基础上修改为有一只闭眼就是闭眼

                more_than_3s = self.more_than_3_seconds()# 判断是否有超过3s的行为
                if more_than_3s:
                    break

                self.failures = 0
            except Exception:
                traceback.print_exc()
                self.failures += 1
                print('jhj [Exception] in detect frame! ', 'self.failures: ', self.failures)
                if self.failures > 30:   # 失败超过30次就默认返回
                    print('break cuase failures > 30')
                    break
            del frame

            # 从帧队列中取出一帧
            frame = self.frame_queue.get()
            frame_idx += self.skip_step  # frame对应的idx
            
            if frame is None:   # 视频结束
                self.result['result']['category'] = 0
                print('Jump out cause video end! ')

            self.detected_frames += 1  # 统计处理过的帧数
            # while next or end


        # 推理结束，记录时间，参考baseline的计时位置，在最后一张图片推理之后
        infer_end_time = time.time()
        duration = int( (infer_end_time - infer_start_time) * 1000 )
        self.result['result']['duration'] = duration


        # 推理结束，结束子线程
        self.io_thread_end = True # 该变量控制子线程结束
        io_thread.join() 

        # 释放视频对象，释放内存
        self.reader.release() # 释放cap对象
        del self.frame_queue # 释放queue
        gc.collect() # gc回收
        print(f'{round(psutil.Process(os.getpid()).memory_full_info().uss / 1024. / 1024, 2)} MB\n') # 打印内存占用
        
        return self.result
    

    def _postprocess(self, data):
        return data
    
    # 初始化/重置计数值
    def reset_CountValue_between_videos(self):
        self.failures = 0    # 失败计数值

        # 四种行为连续计数值
        self.look_around_frame = 0  # 重置这些计数的变量，方便批量测试
        self.eyes_closed_frame = 0
        self.mouth_open_frame = 0
        self.use_phone_frame = 0

        self.cv2_setpos_time = 0  # 统计各部分占的时间
        self.cv2_read_time = 0
        self.onnx_infer_time = 0
        self.tracker_time = 0

        self.detected_frames = 0   # 统计处理的帧的数量，第几帧结束判断的
        self.model1_face = False  # 模型1是否检测到人脸
        self.io_thread_end = False
        return

    # 初始化/重置计数值
    def reset_CountValue_between_frames(self):
        self.is_normal = False  # 一共5种行为, bool类型,表示某种行为有没有出现
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

        return

    def model_1_detect(self, frame):    # 模型1, 图像检测 + 行为判断
        # jhjtime = time.time()
        bbox = self.onnx_model.preprocess(frame)  # bbox是清洗过的数据
        # bbox->[class ,[center-x, center-y, width, height]]
        # time_t = int(np.round((time.time()-jhjtime) * 1000))
        # self.onnx_infer_time += time_t

        # if time_t > 200:
        #     print(f'm1 time: {time_t} ms')

        # print(bbox)

        if len(bbox) > 0:
            for box in bbox:
                # print(box)
                if box[0] == 0:  # 人脸
                    # 这个值只会在不同视频切换重置为False,
                    # 相当于一个视频检测到一次主驾驶员，就一直为true
                    self.model1_face = True
                    # print('人脸')
                elif box[0] == 1:  # 手机
                    self.model1_usePhone = True
                    # print('手机')
                elif box[0] == 2:  # openeye
                    self.model1_openEye += 1
                    self.model1_eyeNum += 1 
                    # print('睁眼')
                elif box[0] == 3:  # closedeye
                    self.model1_closeEye += 1
                    self.model1_eyeNum += 1 
                    # print('闭眼')
                elif box[0] == 4:  # yawn打哈欠
                    self.model1_openMouth = True
                    self.model1_mouthNum += 1
                    # print('打哈欠')
                elif box[0] == 5:  # noyawn 不打哈欠、闭嘴
                    self.model1_closeMouth = True
                    self.model1_mouthNum += 1
                    # print('闭嘴')
                elif box[0] == 6:  # 转头
                    self.model1_face = True
                    self.model1_turned = True
                    # print('转头')
                elif box[0] == 7:  # 低头
                    self.model1_face = True
                    self.model1_turnedDown = True
                    # print('低头')

        else:   # 没有人脸
            self.model1_turned = True
            self.model1_turnedDown = True
            print('Model 1 No Face' )
        return 

    def calculates_timeCount_of_class_v9(self): # 0602 ，1111，在v7的基础上修改为有一只闭眼就是闭眼
        # 多模型规则设计
        # 闭眼
        if self.model1_closeEye > 0: # 在v7的基础上，修改为有一只闭眼就是闭眼
            self.is_closeEye = True
        else:
            self.is_closeEye = False
            
        # 张嘴
        if self.model1_closeMouth or self.model1_mouthNum == 0:  # m1检测到闭嘴，票否
            self.is_openMouth = False
        else:
            self.is_openMouth = True
        
        # 手机
        if self.model1_usePhone:
            self.is_usePhone = True
        # print(self.model1_usePhone)
        
        # 转头
        if self.model1_turned or self.model1_turnedDown: 
            self.is_turned = True


        # 手机
        if self.is_usePhone:
            self.use_phone_frame += self.skip_step
        elif not self.is_usePhone:
            self.use_phone_frame = 0

        # 转头
        if not self.is_turned or (self.is_turned and self.is_usePhone):
            self.look_around_frame = 0
        elif self.is_turned: 
            self.look_around_frame += self.skip_step

        # 张嘴
        if self.is_openMouth:
            self.mouth_open_frame += self.skip_step
        elif not self.is_openMouth:
            self.mouth_open_frame = 0

        # 闭眼        
        if not self.is_closeEye or (self.is_closeEye and self.is_openMouth) or (self.is_closeEye and self.is_turned): # 后续需要替换，缩小优先级范围，将转头影响计时改为低头影响计时
            self.eyes_closed_frame = 0
        elif self.is_closeEye:
            self.eyes_closed_frame += self.skip_step


    def more_than_3_seconds(self):
        more_than_3s = False
        if self.use_phone_frame >= self.frame_3s:
            self.result['result']['category'] = 3
            print('Jump out cuase phone > 3s')
            more_than_3s = True

        elif self.look_around_frame >= self.frame_3s:
            self.result['result']['category'] = 4
            print('Jump out cuase around > 3s')
            more_than_3s = True

        elif self.mouth_open_frame >= self.frame_3s:
            self.result['result']['category'] = 2
            print('Jump out cuase mouth > 3s')
            more_than_3s = True

        elif self.eyes_closed_frame >= self.frame_3s:
            self.result['result']['category'] = 1
            print('Jump out cuase eye > 3s')
            more_than_3s = True
        else:
            self.result['result']['category'] = 0
        
        return more_than_3s

    def print_result_info(self):
        print('file_name:', self.file_name, 'jhj video count:', self.video_count,'height: ', self.height, 'width: ', self.width, 'fps: ', self.fps, 'video_frame_count: ', self.video_frame_count)
        print('cv2_setpos_time', self.cv2_setpos_time, 'cv2_read_time', self.cv2_read_time,'onnx_infer_time', self.onnx_infer_time, 'tracker_time', self.tracker_time)
        return
    
    # IO异步取帧
    def read_frame_worker(self, step:int):
        # self.io_thread_end初始值为False
        while True:
            # 队列为满主动阻塞等待，同时并检查IO线程是否结束
            while self.frame_queue.full():
                if self.io_thread_end:  # 阻塞期间检查IO线程是否应该结束
                    return
                time.sleep(0.05)    # 人为阻塞50ms

            # 取帧
            ret, frame = self.reader.grab_forword_read(step)            
            if ret:
                self.frame_queue.put(frame)
            else:
                self.frame_queue.put(None) # 读出来的一帧为空，意为视频结束
                break
        return
