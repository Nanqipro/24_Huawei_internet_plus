import os
import cv2
import numpy as np
import psutil
import shutil

'''推理部分'''
# 查看内存占用
def show_memory_info(hint):
    pid = os.getpid()  # 进程ID
    p = psutil.Process(pid)  # 返回进程对象,不传 pid 默认会获取当前的pid
    info = p.memory_full_info()  # pfullmem 对象
    memory = round(info.uss / 1024. / 1024, 2)
    print(f'{hint} memory used: {memory} MB')



'''score'''
def get_video_time(file_path):
    # 打开视频文件
    cap = cv2.VideoCapture(file_path)
    # 获取视频的帧率
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # 获取视频的总帧数
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 计算视频时长，frame_count/fps的单位是秒, 转换成ms
    duration = frame_count / fps * 1000
    # print(f"视频时长是 {duration} 秒。")
    # 释放资源
    cap.release()

    return duration

# sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 传入一个文件名，获取真实类别
def get_true_label(filename: str):
    # day_man_001_10_1.mp4
    # 环境_性别_被采集人员编号_动作编码_序号
    text = filename.split('_')

    if text[3] == '10':
        ture_lebel = 1  # 闭眼（的正样本，负样本划归正常驾驶）
    elif text[3] == '20':
        ture_lebel = 2  # 打哈欠（的正样本，负样本划归正常驾驶）
    elif text[3] == '30':
        ture_lebel = 3  # 打电话（的正样本，负样本划归正常驾驶）
    elif text[3] == '40':
        ture_lebel = 4  # 左顾右盼（的正样本，负样本划归正常驾驶）
    else:
        ture_lebel = 0  # 正常驾驶或其他四类的负样本，都归为正常驾驶

    return ture_lebel

'''onnx_detect'''
def xyxy2xywh(xyxy):
    x1 = xyxy[0]
    y1 = xyxy[1]
    x2 = xyxy[2]
    y2 = xyxy[3]

    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    xywh = [x,y,w,h]
    return xywh

# 计算IOU
def iouxyxy(xyxy_a, xyxy_b):
    # get area of a
    area_a = (xyxy_a[2] - xyxy_a[0]) * (xyxy_a[3] - xyxy_a[1]) # h*w
    # get area of b
    area_b = (xyxy_b[2] - xyxy_b[0]) * (xyxy_b[3] - xyxy_b[1]) # h*w

    # get left top x of IoU
    inter_x1 = np.maximum(xyxy_a[0], xyxy_b[0])
    # get left top y of IoU
    inter_y1 = np.maximum(xyxy_a[1], xyxy_b[1])
    # get right bottom of IoU
    inter_x2 = np.minimum(xyxy_a[2], xyxy_b[2])
    # get right bottom of IoU
    inter_y2 = np.minimum(xyxy_a[3], xyxy_b[3])

    # get width of IoU
    inter_w = np.maximum(inter_x2 - inter_x1, 0) 
    # get height of IoU
    inter_h = np.maximum(inter_y2 - inter_y1, 0) 

    # get area of IoU
    area_inter = inter_w * inter_h
    # get overlap ratio between IoU and all area
    iou = area_inter / (area_a + area_b - area_inter)

    return iou

# 清洗检测到的对象，只保留与驾驶员相关的    
def func_clean(result: list, frame_area, height, width):  # 列表中的一个元素 -> [cls_id, [x, y, w, h]]
    # list为空
    if len(result) <= 0:
        return []
    
    # 结果列表，存放清洗后的结果
    clean_result = []

    # 通过面积确定哪个是驾驶员
    maxArea = 0
    maxIndex = -1
    driver_index = -1
    for i, r in enumerate(result):
        # 如果是人脸(1和6都是脸，脸和转头、低头)
        if r[0] == 0 or r[0] == 6 or r[0] == 7:
            area = r[1][2] * r[1][3] # 分别为xywh，此处为wh
            if area > maxArea and area / frame_area > 0.05:
                maxArea = area
                maxIndex = i
    driver_index = maxIndex # 通过面积确定哪个是驾驶员idx

    # 根据驾驶员的idx和box信息，来过滤其他五个类
    if driver_index != -1:
        # print('确定驾驶员的脸位置为', result[driver_index])
        drive_center_x = result[driver_index][1][0]     # center-x
        drive_center_y = result[driver_index][1][1]     # center-y
        drive_face_width = result[driver_index][1][2]        # w
        drive_face_height = result[driver_index][1][3]       # h
        # drive_face_width = drive_face_width * 1.1  # 人脸区域横向扩展
        # 驾驶员人脸位置
        d_x1 = np.maximum(drive_center_x - drive_face_width / 2, 0)
        d_y1 = np.maximum(drive_center_y - drive_face_height / 2, 0)
        d_x2 = np.minimum(drive_center_x + drive_face_width / 2, width)
        d_y2 = np.minimum(drive_center_y + drive_face_height / 2, height)
        face_box = [d_x1, d_y1, d_x2, d_y2]  # 中心点坐标和宽高
        
        # 胸前可能出现手机的区域
        p_x1 = np.maximum(drive_center_x - drive_face_width, 0)
        p_y1 = (height + d_y2) / 2
        p_x2 = np.minimum(drive_center_x + drive_face_width, width)
        p_y2 = height
        phone_area = [p_x1, p_y1, p_x2, p_y2]

        # 根据驾驶员人脸判断其他类别是否属于驾驶员
        for i, r in enumerate(result):
            x = r[1][0]
            y = r[1][1]
            w = r[1][2]
            h = r[1][3]
            x1 = x - w/2
            y1 = y - h/2
            x2 = x + w/2
            y2 = y + h/2
            obj_box = [x1, y1, x2, y2]
            iou = iouxyxy(obj_box, face_box)
            if (r[0] == 0 or r[0] == 6 or r[0] == 7) and iou > 0.5: # 正脸、转头、低头
                clean_result.append(r) # 确保如果有转头，转头不会被face顶掉
                # print(f'{result[driver_index]}和{r}的iou为{iou}, 因为条件2: if (r[0] == 0 or r[0] == 6 or r[0] == 7) and iou > 0.5被添加')
            elif (r[0] == 2 or r[0] == 3 or r[0] == 4 or r[0] == 5) and iou > 0:       # 手机、眼睛、嘴巴
                clean_result.append(r) # 眼睛、嘴巴加入list
                # print(f'{result[driver_index]}和{r}的iou为{iou}, 因为条件2: elif iou > 0被添加')
            elif r[0] == 1:
                iou2 = iouxyxy(obj_box, phone_area)
                if iou>0 or iou2>0:
                    clean_result.append(r) # 将手机

                        
    return clean_result  # 返回人脸面积最大的那个人脸

# 用于测试脚本，将一批文件复制到另一个位置

def copyFile(input_dir, file_name, output_dir):
    '''
    input_dir: 输入路径,
    file_list: 输入路径下的文件名, 
    output_dir: 输出路径, 将文件复制到哪里
    '''
    in_path = os.path.join(input_dir, file_name)
    out_path = os.path.join(output_dir, file_name)
    shutil.copyfile(in_path, out_path)

