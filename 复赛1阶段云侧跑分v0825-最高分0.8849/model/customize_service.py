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
import argparse
# from sklearn.metrics import f1_score

from model1.onnx_detect import ONNX_engine
import our_tool
from our_videoReader import VideoReader

# from model_service.pytorch_model_service import PTServingBaseService # 提交
# class fatigue_driving_detection(PTServingBaseService): # 提交


class fatigue_driving_detection():  # 本地
    def __init__(self, model_name, model_path):
        self.onnx_model = ONNX_engine(weights_path='v4_c7_320.onnx', size=320)   # 提交！！！！看清楚权重
        self.model_name = model_name
        self.model_path = model_path
        self.capture = ''

        self.width = 1920
        self.height = 1080
        self.fps = 30
        self.frameNum_3s = self.fps * 3        # 3s有几帧
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

    # 前处理：接收byteIO对象并存成mp4文件
    def _preprocess(self, data):
        # data: {'input_video': OrderedDict([('day_man_101_41_6.mp4', <_io.BytesIO object at 0x7fd95e91c770>)])}
        self.video_count += 1
        for k, v in data.items():
            for file_name, bytesIO_object in v.items():
                print('file_name:', file_name,
                      'jhj video count:', self.video_count)
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
        # self.result = {"result": {"category": 0, "duration": 6000}}
        self.result = {"result": {"duration": 6000, "drowsy": []}}
        self.temp_list = []
        self.temp_action = {"periods": [3, 7], "category": 1}

        self.timestamp_start_lookAround = 0         # 初始化记录时间戳（起、止时间）的变量
        self.timestamp_start_closeEye = 0
        self.timestamp_start_openMouth = 0
        self.timestamp_start_usePhone = 0

        self.timestamp_end_lookAround = 0
        self.timestamp_end_closeEye = 0
        self.timestamp_end_openMouth = 0
        self.timestamp_end_usePhone = 0

        self.lookAround_MoreThan3s = False
        self.closeEye_MoreThan3s = False
        self.openMouth_MoreThan3s = False
        self.usePhone_MoreThan3s = False

        self.reset_CountValue_between_videos()          # 初始化，重置不同视频间的计数值，便于批量测试

        self.reader = VideoReader(self.capture)         # 初始化，视频capture
        if self.reader.ret == False:
            return self.result

        self.height, self.width, self.fps, self.video_totalFrameNum = self.reader.get_capInfo()   # 初始化，获取视频信息

        self.frameNum_3s = int(self.fps * 3)                    # 3s的帧数
        self.skip_step = 6                                      # 固定帧数间隔

        deal_frame_ps = 3                                      # 初始化，设置跳帧，每秒处理几帧
        self.skip_step = int(self.fps/float(deal_frame_ps))     # 按照每秒取多少帧来处理，来确定step

        # 初始化，帧队列
        self.frame_queue = queue.Queue(maxsize=3)
        io_thread = threading.Thread(target=self.read_frame_worker, args=(
            self.skip_step, ))    # 初始化，创建一个子线程用于取帧，异步IO
        # 守护线程, 随进程结束而结束
        io_thread.daemon = True
        io_thread.start()

        frame = self.frame_queue.get()              # 初始化，第一个视频帧，从帧队列中获取一帧
        self.nowFrameIdx = self.skip_step           # 当前帧（frame）是第几帧。视频开始的时候，取出的第一帧为一个跳帧间隔处的帧

        # 模型推理部分
        infer_start_time = time.time()              # 开始记录推理时间
        while frame is not None:
            self.reset_CountValue_between_frames()          # 重置帧间的计数值
            print(f'start dealing nowFrameIdx : {self.nowFrameIdx}' + ' ' + f'timestamp : {self.getTimestamp()}')
            try:
                # 画面剪裁，主驾驶位； [h, w, 通道]
                frame = frame[:, int(self.width * 0.33): int(self.width), :]
                # print(frame.shape[0],frame.shape[1],frame.shape[2])

                self.model_1_detect(frame)              # 模型1，图像检测
                self.judge_action()                     # 判断该帧中的行为
                self.deal_action_count()
                self.failures = 0

            except Exception:
                traceback.print_exc()
                self.failures += 1
                print('jhj [Exception] in detect frame! ',
                      'self.failures: ', self.failures)
                if self.failures > 30:                  # 失败超过30次就默认返回
                    print('break cuase failures > 30')
                    break

            del frame
            self.detected_frames += 1               # 处理过的帧数+1
            frame = self.frame_queue.get()          # 从帧队列中取出新的一帧
            self.nowFrameIdx += self.skip_step      # 新的一帧对应的位置
            # while next or end

        # 推理结束，记录时间
        infer_end_time = time.time()
        duration = int((infer_end_time - infer_start_time) * 1000)
        self.result['result']['duration'] = duration
        self.result["result"]["drowsy"] = self.temp_list

        # 推理结束，结束子线程
        self.io_thread_end = True  # 该变量控制子线程结束
        io_thread.join()

        # 释放视频对象，释放内存
        self.reader.release()  # 释放cap对象
        del self.frame_queue  # 释放queue
        

        self.print_result_info()        # 打印结果信息
        return self.result

    def _postprocess(self, data):
        return data

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

    # 模型1，图像检测，检测出现的行为
    def model_1_detect(self, frame):
        jhjtime = time.time()
        # bbox是清洗过的数据    # bbox->[class ,[center-x, center-y, width, height]]
        bbox = self.onnx_model.preprocess(frame)
        time_t = int(np.round((time.time()-jhjtime) * 1000))
        self.onnx_infer_time += time_t
        if time_t > 200:
            print(f'm1 time: {time_t} ms')
        # print(bbox)

        if len(bbox) > 0:
            for box in bbox:
                # print(box)
                if box[0] == 0:                     # 人脸
                    self.model1_face = True         # 这个值只会在不同视频切换重置为False。相当于一个视频检测到一次主驾驶员，就一直为true
                    # print('人脸')
                elif box[0] == 1:                   # 手机
                    self.model1_usePhone = True
                    # print('手机')
                elif box[0] == 2:                   # openeye
                    self.model1_openEye += 1
                    self.model1_eyeNum += 1
                    # print('睁眼')
                elif box[0] == 3:                   # closedeye
                    self.model1_closeEye += 1
                    self.model1_eyeNum += 1
                    # print('闭眼')
                elif box[0] == 4:                   # yawn打哈欠
                    self.model1_openMouth = True
                    self.model1_mouthNum += 1
                    # print('打哈欠')
                elif box[0] == 5:                   # noyawn 不打哈欠、闭嘴
                    self.model1_closeMouth = True
                    self.model1_mouthNum += 1
                    # print('闭嘴')
                elif box[0] == 6:                   # 转头
                    self.model1_face = True
                    self.model1_turned = True
                    # print('转头')
                elif box[0] == 7:                   # 低头，不识别低头的模型也不影响，不会出现7，该条件判断不成立
                    self.model1_face = True
                    self.model1_turnedDown = True
                    # print('低头')
        else:                                       # 没有人脸
            self.model1_turned = True
            self.model1_turnedDown = True
            print('Model 1 No Face')
        return

    def judge_action(self):         # 0602 ，1111，在v7的基础上修改为有一只闭眼就是闭眼

        if self.model1_closeEye > 0:                                # 闭眼，在v7的基础上，修改为有一只闭眼就是闭眼
            self.is_closeEye = True
        else:
            self.is_closeEye = False

        if self.model1_closeMouth or self.model1_mouthNum == 0:     # 张嘴，m1检测到闭嘴，票否
            self.is_openMouth = False
        else:
            self.is_openMouth = True
        # if self.model1_openMouth:
        #     self.is_openMouth = True
        # else: 
        #     self.is_openMouth = False
        

        if self.model1_usePhone:                                    # 手机
            self.is_usePhone = True

        if self.model1_turned or self.model1_turnedDown:            # 转头
            self.is_turned = True

    def deal_action_count(self):
        nowTimestamp = self.getTimestamp()                  # 获取当前时间戳
        half_step_timestamp = int(1000.0 * self.skip_step / 2 / self.fps)
        self.videoWillEnd = False           # 下一帧视频将会结束
        if self.nowFrameIdx + self.skip_step >= self.video_totalFrameNum:       # 下次跳帧超出总帧数，代表视频结尾
            self.videoWillEnd = True

        # 手机
        if self.is_usePhone:            # 检测到手机
            print(f'((3) {self.use_phone_frame})')
            self.use_phone_frame += self.skip_step
            if self.use_phone_frame == self.skip_step:              # 动作刚开始，记录 开始时间
                self.timestamp_start_usePhone = nowTimestamp - half_step_timestamp
            elif self.use_phone_frame >= self.frameNum_3s:          # 判断某个行为是否超过3s，若超过置标志位为True
                self.usePhone_MoreThan3s = True
            
            if self.videoWillEnd and self.usePhone_MoreThan3s:              # 视频结束 and 行为超过3s
                self.timestamp_end_usePhone = nowTimestamp - half_step_timestamp  # 记录 结束时间
                self.temp_action["periods"][0] = self.timestamp_start_usePhone
                self.temp_action["periods"][1] = self.timestamp_end_usePhone
                self.temp_action["category"] = 3
                self.temp_list.append(copy.deepcopy(self.temp_action))
                self.clear_between_action()                
        else:                           # 没检测到手机
            self.use_phone_frame = 0
            if self.usePhone_MoreThan3s:              # 这个行为结束的时候超过3s
                self.timestamp_end_usePhone = nowTimestamp - half_step_timestamp # 记录 结束时间
                self.temp_action["periods"][0] = self.timestamp_start_usePhone
                self.temp_action["periods"][1] = self.timestamp_end_usePhone
                self.temp_action["category"] = 3
                self.temp_list.append(copy.deepcopy(self.temp_action))
                self.clear_between_action()



        # 转头
        if not self.is_turned or (self.is_turned and self.is_usePhone):
            self.look_around_frame = 0
            if self.lookAround_MoreThan3s:        # 这个行为结束的时候超过3s
                self.timestamp_end_lookAround = nowTimestamp - half_step_timestamp
                self.temp_action["periods"][0] = self.timestamp_start_lookAround
                self.temp_action["periods"][1] = self.timestamp_end_lookAround
                self.temp_action["category"] = 4
                self.temp_list.append(copy.deepcopy(self.temp_action))
                self.clear_between_action()

        elif self.is_turned:
            print(f'((4) {self.look_around_frame})')
            self.look_around_frame += self.skip_step
            if self.look_around_frame == self.skip_step:  # 初次出现某个危险行为，计数值 = 一个跳帧间隔
                self.timestamp_start_lookAround = nowTimestamp - half_step_timestamp
            elif self.look_around_frame >= self.frameNum_3s:
                self.lookAround_MoreThan3s = True
            
            if self.videoWillEnd and self.lookAround_MoreThan3s:        # 这个行为结束的时候超过3s
                self.timestamp_end_lookAround = nowTimestamp - half_step_timestamp
                self.temp_action["periods"][0] = self.timestamp_start_lookAround
                self.temp_action["periods"][1] = self.timestamp_end_lookAround
                self.temp_action["category"] = 4
                self.temp_list.append(copy.deepcopy(self.temp_action))
                self.clear_between_action()

        # 张嘴
        if self.is_openMouth:
            print(f'((2) {self.mouth_open_frame})')
            self.mouth_open_frame += self.skip_step
            if self.mouth_open_frame == self.skip_step:
                self.timestamp_start_openMouth = nowTimestamp - half_step_timestamp
            elif self.mouth_open_frame >= self.frameNum_3s:
                self.openMouth_MoreThan3s = True
            
            if self.videoWillEnd and self.openMouth_MoreThan3s:         # 这个行为结束的时候超过3s
                self.timestamp_end_openMouth = nowTimestamp - half_step_timestamp
                self.temp_action["periods"][0] = self.timestamp_start_openMouth
                self.temp_action["periods"][1] = self.timestamp_end_openMouth
                self.temp_action["category"] = 2
                self.temp_list.append(copy.deepcopy(self.temp_action))
                self.clear_between_action()
        elif not self.is_openMouth:
            self.mouth_open_frame = 0
            if self.openMouth_MoreThan3s:         # 这个行为结束的时候超过3s
                self.timestamp_end_openMouth = nowTimestamp - half_step_timestamp
                self.temp_action["periods"][0] = self.timestamp_start_openMouth
                self.temp_action["periods"][1] = self.timestamp_end_openMouth
                self.temp_action["category"] = 2
                self.temp_list.append(copy.deepcopy(self.temp_action))
                self.clear_between_action()



        # 闭眼
        # 后续需要替换，缩小优先级范围，将转头影响计时改为低头影响计时
        if not self.is_closeEye or (self.is_closeEye and self.is_openMouth) or (self.is_closeEye and self.is_turned):
            self.eyes_closed_frame = 0
            if self.closeEye_MoreThan3s:         # 这个行为结束的时候超过3s
                self.timestamp_end_closeEye = nowTimestamp - half_step_timestamp
                self.temp_action["periods"][0] = self.timestamp_start_closeEye 
                self.temp_action["periods"][1] = self.timestamp_end_closeEye 
                self.temp_action["category"] = 1
                self.temp_list.append(copy.deepcopy(self.temp_action))
                self.clear_between_action()
        elif self.is_closeEye:
            print(f'((1) {self.eyes_closed_frame})')
            self.eyes_closed_frame += self.skip_step
            if self.eyes_closed_frame == self.skip_step:
                self.timestamp_start_closeEye = nowTimestamp - half_step_timestamp
            elif self.eyes_closed_frame >= self.frameNum_3s:
                self.closeEye_MoreThan3s = True
            
            if self.videoWillEnd and self.closeEye_MoreThan3s:         # 这个行为结束的时候超过3s
                self.timestamp_end_closeEye = nowTimestamp - half_step_timestamp
                self.temp_action["periods"][0] = self.timestamp_start_closeEye
                self.temp_action["periods"][1] = self.timestamp_end_closeEye
                self.temp_action["category"] = 1
                self.temp_list.append(copy.deepcopy(self.temp_action))
                self.clear_between_action()
        

    def clear_between_action(self):

        self.timestamp_start_lookAround = 0         # 重置时间戳（起、止时间）的变量
        self.timestamp_start_closeEye = 0
        self.timestamp_start_openMouth = 0
        self.timestamp_start_usePhone = 0

        self.timestamp_end_lookAround = 0
        self.timestamp_end_closeEye = 0
        self.timestamp_end_openMouth = 0
        self.timestamp_end_usePhone = 0

        self.lookAround_MoreThan3s = False        # 超过3s的标记
        self.closeEye_MoreThan3s = False
        self.openMouth_MoreThan3s = False
        self.usePhone_MoreThan3s = False

        self.look_around_frame = 0                      # 每个行为检测出来以后，重置连续帧数计数值
        self.eyes_closed_frame = 0
        self.mouth_open_frame = 0
        self.use_phone_frame = 0

        return

    def getTimestamp(self):             # 获取以ms为单位的时间戳
        # 第几帧/fps*1000 即为时间戳，单位ms
        return int(1000.0 * self.nowFrameIdx / self.fps)

    def print_result_info(self):
        gc.collect()
        print(f'{round(psutil.Process(os.getpid()).memory_full_info().uss / 1024. / 1024, 2)} MB\n')                # 打印内存占用 

        print('file_name:', self.file_name, '; jhj video count:', self.video_count, '; height: ', self.height,
              '; width: ', self.width, '; fps: ', self.fps, '; video_frame_count: ', self.video_totalFrameNum)
        print(f'detected_frames: {self.detected_frames} Done; onnx_infer_time: {self.onnx_infer_time} ms')
        print( self.result)      # 处理过的帧数
        return

    # IO异步取帧
    def read_frame_worker(self, step: int):
        while True:
            while self.frame_queue.full():          # 队列为满主动阻塞等待，同时并检查IO线程是否结束
                if self.io_thread_end:              # self.io_thread_end初始值为False。阻塞期间检查IO线程是否应该结束
                    return
                time.sleep(0.01)                    # 人为阻塞
            ret, frame = self.reader.grab_forword_read(step)      # 取帧
            if ret:
                self.frame_queue.put(frame)
            else:
                self.frame_queue.put(None)          # 读出来的一帧为空，意为视频结束
                break
        return


def run(opt):

    if opt.savetxt:
        if not os.path.exists('run'):               # 路径不存在则新建
            os.makedirs('run')
        save_stdout = sys.stdout                    # 保存标准输出流
        info_output = open(os.path.join('run', opt.output_txtname), "w")
        sys.stdout = info_output

    detection = fatigue_driving_detection("model name", opt.pt_weights)
    detection.onnx_model = ONNX_engine(
        weights_path=opt.weights, size=opt.img_size)

    # 调参 眼睛ear阈值、嘴巴mar阈值、每三帧取一帧
    detection.skip_step = opt.skip_step

    filenames = os.listdir(opt.test_video)          # 将文件夹转换成一个列表，元素为文件名
    filenames.sort()                                # 对列表进行排序, 元素是字符串时，默认按照首字母排序

    y_true = []
    y_pred = []

    all_deal_frame = 0

    all_time = 0
    all_detect_time = 0

    Total_time_score = 0
    video_num = len(filenames)  # 视频段数
    count = 0

    print('Wait...\ncategory: 0正常驾驶 1闭眼 2打哈欠 3打电话 4左顾右盼\n')

    filter_class_10 = str(opt.filter_class)+'0'
    filter_class_11 = str(opt.filter_class)+'1'

    filter_filename = []

    if opt.filter:
        # 统计一下满足筛选条件的视频文件数
        video_num = 0
        for filename in filenames:
            text = filename.split('_')
            if text[3] != filter_class_10 and text[3] != filter_class_11:
                continue
            video_num += 1

    for filename in filenames:

        if opt.filter:

            # 添加筛选条件，例如只跑类别标签为10 11 的视频
            text = filename.split('_')
            if text[3] != filter_class_10 and text[3] != filter_class_11:
                continue

        file_path = os.path.join(opt.test_video, filename)  # 拼接文件路径

        detection.capture = file_path
        result = detection._inference(1)

        category = result.get('result').get('category')
        infer_duration = result.get('result').get('duration')

        y_true.append(our_tool.get_true_label(filename))  # 从文件名获取真实标签
        filter_filename.append(filename)
        y_pred.append(category)  # 预测出来的标签

        # 计算推理时长得分
        video_time = round(our_tool.get_video_time(
            file_path))  # 计算单个视频的视频时长（单位为ms）
        time_score = our_tool.sigmoid(
            video_time / (infer_duration * opt.time_slow))
        Total_time_score += time_score

        all_deal_frame += detection.detected_frames
        all_time += infer_duration

        all_detect_time += detection.onnx_infer_time

        count += 1

        print(count, '/', video_num, filename, result, '; y_true',
              y_true[-1], '; y_pred', y_pred[-1], '; video_time(ms)', video_time, '; time_score', round(time_score, 3))
        print('\n\n')

        # 打印到终端
        if opt.savetxt and opt.print2terminal:
            sys.stdout = save_stdout  # 恢复标准输出流
            print(count, '/', video_num, filename, result, '; y_true',
                  y_true[-1], '; y_pred', y_pred[-1], '; video_time(ms)', video_time, '; time_score', round(time_score, 3))
            print('\n\n')
            sys.stdout = info_output

        result = detection._postprocess(result)

        info_output.flush()  # 立刻写入文件，不等缓存

    # 计算各个分类的加权F1_score
    # F1_score = f1_score(y_true, y_pred, average='weighted') * 0.7
    # 视频的时间得分，乘权重0.3。（* 100 / video_num这个操作是为了不是100段视频时，总得分不为100）
    Total_time_score = Total_time_score / video_num * 0.3

    # 总得分
    # Total_socre = F1_score + Total_time_score
    print('\n', opt.test_video)
    # print("Total score", round(Total_socre, 3), "F1_score", round(F1_score,4), "Total_time_score", round(Total_time_score,3), '\n')

    # 打印到终端
    if opt.savetxt and opt.print2terminal:
        sys.stdout = save_stdout  # 恢复标准输出流
        print('\n', opt.test_video)
        # print("Total score", round(Total_socre, 3), "F1_score", round(F1_score,4), "Total_time_score", round(Total_time_score,3), '\n')

        count_err = 0
        for i in range(len(y_true)):
            if y_true[i] != y_pred[i]:
                print(i+1, '/', video_num,
                      filter_filename[i], '; y_true', y_true[i], '; y_pred', y_pred[i])
                count_err += 1

        print('count_err', count_err, '/', video_num)
        print('all_deal_frame', all_deal_frame)
        print('all_time', all_time)
        print('all_detect_time', all_detect_time)

        sys.stdout = info_output

    # 输出预测错误的视频
    count_err = 0

    err_copy = opt.err_copy
    if err_copy:
        err_dir = opt.err_outdir
        if not os.path.exists(err_dir):
            os.makedirs(err_dir)

    right_copy = opt.right_copy
    if right_copy:
        right_dir = opt.right_outdir
        if not os.path.exists(right_dir):
            os.makedirs(right_dir)

    in_dir = opt.test_video

    # 统计分布
    # distribute_dict = {'0':{'0':0, '1':0, '2':0, '3':0, '4':0},
    #                    '1':{'0':0, '1':0, '2':0, '3':0, '4':0},
    #                    '2':{'0':0, '1':0, '2':0, '3':0, '4':0},
    #                    '3':{'0':0, '1':0, '2':0, '3':0, '4':0},
    #                    '4':{'0':0, '1':0, '2':0, '3':0, '4':0}}
    # for i in range(len(y_true)):
    #     if y_true[i] != y_pred[i]:
    #         distribute_dict[str(y_true[i])][str(y_pred[i])] += 1
    #         print(i+1, '/', video_num, filter_filename[i], '; y_true', y_true[i], '; y_pred', y_pred[i])
    #         if err_copy:
    #             our_tool.copyFile(in_dir, filter_filename[i], err_dir)
    #         count_err += 1
    #     else:
    #         distribute_dict[str(y_true[i])][str(y_pred[i])] += 1
    #         if right_copy:
    #             our_tool.copyFile(in_dir, filter_filename[i], right_dir)

    print('count_err', count_err, '/', video_num)
    print('all_deal_frame', all_deal_frame)
    print('all_time', all_time)
    print('all_detect_time', all_detect_time)

    # print('Distribute Dict: ', distribute_dict)

    if opt.savetxt:
        sys.stdout = save_stdout  # 恢复标准输出流
        info_output.close()


if __name__ == "__main__":

    # 使用argparse打包参数，便于从命令行修改参数或者使用shell脚本进行测试
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='v4_c7_320.onnx', help='onnx权重路径')
    parser.add_argument('--test-video', type=str,
                        default='D:\\BaiduSyncdisk\\科研\\华为云互联网+\\数据集\\疲劳驾驶检测视频数据\\video2044\\newdataset_video', help='测试视频路径')
    parser.add_argument('--savetxt', type=bool, default=True, help='是否保存输出txt')
    parser.add_argument('--output-txtname', type=str,
                        default='output0824.txt', help='输出日志文件路径')
    parser.add_argument('--img-size', type=int, default=320, help='推理时使用的图片大小')
    parser.add_argument('--eye', type=float, default=0.18, help='闭眼阈值')
    parser.add_argument('--mouth', type=float, default=0.55, help='张嘴阈值')
    parser.add_argument('--skip-step', type=int, default=6, help='基础跳帧间隔')
    parser.add_argument('--print2terminal', type=bool,
                        default=False, help='是否将中间过程打印到终端')
    parser.add_argument('--time-slow', type=float,
                        default=2.5, help='平台时间比本地时间慢多少, 模仿真实时间得分')
    parser.add_argument('--limit-cpu', type=bool,
                        default=False, help='是否限制cpu, 但不是绝对的限制, 略微有点鸡肋')
    parser.add_argument('--limit-cpu-core', type=list,
                        default=[21, 22], help='若限制cpu, 用一个列表表示要绑定的cpu核数')
    parser.add_argument('--filter', type=bool,
                        default=False, help='是否筛选指定类别的视频进行推理测试')
    parser.add_argument('--filter-class', type=int,
                        default=1, help='输入1、2、3、4来表示删选哪个类')
    parser.add_argument('--pt-weights', type=str,
                        default='our_yolov7t_mbnv3_1.pt', help='pt权重路径')
    parser.add_argument('--err-copy', type=bool,
                        default=False, help='是否将，预测错误的视频被单独复制到另外的文件夹下')
    parser.add_argument('--err-outdir', type=str,
                        default='err_video/test', help='预测错误的视频被单独复制到这个文件夹下')
    parser.add_argument('--right-copy', type=bool,
                        default=False, help='是否将，预测正确的视频被单独复制到另外的文件夹下')
    parser.add_argument('--right-outdir', type=str,
                        default='right_video/test', help='预测正确的视频被单独复制到这个文件夹下')
    opt = parser.parse_args()

    # 测试
    run(opt)
