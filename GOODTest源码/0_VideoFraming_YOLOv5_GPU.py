import cv2
import torch
import csv
# import mysql.connector
# from mysql.connector import Error

# 使用YOLOv5目标检测算法对视频逐帧目标检测和并提取关键帧中的对象
# 此算法基于pytorch框架，对于检测结果构建索引并存入csv文件中和MySQL数据库中
# 优化方法：1、目标检测算法运用GPU进行加速计算，大大降低了训练耗时，提高了运算效率
#         2、使用批处理和预编译语句优化数据库操作


# 加载预训练的YOLOv5模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


# 检查GPU是否可用并加载模型到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)


# 定义处理视频的函数
# 导入OpenCV库，用于视频处理
def process_video(video_path):
    """
    处理视频文件。

    该函数尝试打开指定路径的视频文件，并进行后续处理。如果无法打开视频文件，则打印错误信息并退出函数。

    参数:
    video_path: 字符串，表示视频文件的路径。

    返回值:
    无
    """
    # 使用OpenCV的VideoCapture函数打开视频文件
    cap = cv2.VideoCapture(video_path)
    # 检查视频文件是否成功打开
    if not cap.isOpened():
        # 如果视频文件无法打开，则打印错误信息并返回
        print("Error: Could not open video.")
        return


    # 初始化帧索引和检测结果列表
    frame_index = 0
    detections = []

    # 不断读取视频帧直到结束
    while True:
        # 从视频捕获设备中读取一帧
        ret, frame = cap.read()
        # 如果帧读取失败，退出循环
        if not ret:
            break
        # 将BGR格式的帧转换为RGB格式，以适应模型输入
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 使用模型对当前帧进行检测
        results = model(frame_rgb)
        # 将检测结果转换为Pandas DataFrame格式
        results_data = results.pandas().xyxy[0]
        # 将当前帧的索引和检测结果添加到列表中
        detections.append({
            'frame_index': frame_index,
            'detections': results_data.to_dict(orient='records')
        })
        # 增加帧索引
        frame_index += 1

    # 释放视频捕获设备
    cap.release()
    # 返回所有的检测结果
    return detections

# 构建索引（创建一个字典来存储每个对象标签及其出现的帧和时间码）
def build_index(detections, fps):
    """
    构建一个索引，将检测到的对象按时间顺序索引。

    参数:
    detections: 包含多个检测结果的列表，每个检测结果包含一个帧索引和一组检测到的对象。
    fps: 帧率，用于将帧索引转换为时间戳。

    返回:
    一个字典，键为对象名称，值为一个列表，列表中每个元素包含帧索引、时间戳和对象的置信度。
    """
    # 初始化一个空的索引字典
    index = {}
    # 遍历所有的检测结果
    for detection in detections:
        # 提取当前检测结果的帧索引
        frame_index = detection['frame_index']
        # 计算对应的时间戳
        time_stamp = frame_index / fps  # 计算时间戳
        # 遍历当前帧中检测到的所有对象
        for det in detection['detections']:
            # 提取对象的标签
            label = det['name']
            # 如果当前标签尚未出现在索引中，则初始化一个空的列表
            if label not in index:
                index[label] = []
            # 将当前对象的帧索引、时间戳和置信度添加到对应的列表中
            index[label].append({'frame_index': frame_index, 'time_stamp': time_stamp, 'confidence': det['confidence']})
    # 返回构建完成的索引字典
    return index

# 将结果保存入csv文件中
def save_index_to_csv(index, csv_file_path):
    """
    将索引数据保存到CSV文件中。

    参数:
    index: 字典类型，包含标签、帧索引、时间戳和置信度等信息。
    csv_file_path: 字符串类型，表示CSV文件的路径。

    该函数打开指定的CSV文件，以写入模式创建一个writer对象，然后遍历索引数据，
    将每条数据写入CSV文件中，每行数据包括标签、帧索引、时间戳和置信度。
    """
    # 以写入模式打开CSV文件，并配置不生成额外的空行
    with open(csv_file_path, mode='w', newline='') as file:
        # 创建CSV写入器
        writer = csv.writer(file)
        # 写入CSV文件的表头
        writer.writerow(['Label', 'Frame Index', 'Time Stamp', 'Confidence'])

        # 遍历索引字典中的每个标签及其对应的数据条目
        for label, entries in index.items():
            for entry in entries:
                # 将每条数据的各个字段写入CSV文件的一行
                # 使用f-string格式化浮点数，保留两位小数
                writer.writerow(
                    [label, entry['frame_index'], f"{entry['time_stamp']:.2f}", f"{entry['confidence']:.2f}"])

# 连接到MySql数据库
# def connect_to_mysql():
#     try:
#         connection = mysql.connector.connect(
#             host='localhost',
#             user='root',
#             password='8729512929abc',
#             database='video_analysis'
#         )
#         if connection.is_connected():
#             db_Info = connection.get_server_info()
#             print("Connected to MySql Server version ", db_Info)
#             return connection
#     except Error as e:
#         print("Error while connecting to MySql", e)

# 插入检测数据到数据库
# def insert_objects(index):
#     connection = connect_to_mysql()
#     if connection is not None:
#         cursor = connection.cursor()
#         insert_query = """
#         INSERT INTO objects (label, frame_index, time_stamp, confidence)
#         VALUES (%s, %s, %s, %s)
#         """
#         for label, entries in index.items():
#             for entry in entries:
#                 data_tuple = (label, entry['frame_index'], entry['time_stamp'], entry['confidence'])
#                 cursor.execute(insert_query, data_tuple)
#         connection.commit()
#         print("Data inserted successfully")
#         cursor.close()
#         connection.close()

# 改进后使用批处理和预编译语句优化数据库操作
# （逐条执行插入操作效率低下，尤其是在处理大量数据时。所以通过使用预编译语句和批处理来优化这一过程。）
# def insert_objects(index):
#     connection = connect_to_mysql()
#     if connection is not None:
#         cursor = connection.cursor(prepared=True)
#         insert_query = """
#         INSERT INTO objects (label, frame_index, time_stamp, confidence)
#         VALUES (%s, %s, %s, %s)
#         """
#         batch_data = []
#         for label, entries in index.items():
#             for entry in entries:
#                 data_tuple = (label, entry['frame_index'], entry['time_stamp'], entry['confidence'])
#                 batch_data.append(data_tuple)
#                 if len(batch_data) >= 1000:  # 批量插入，每批1000条
#                     cursor.executemany(insert_query, batch_data)
#                     batch_data = []
#         if batch_data:  # 插入剩余的数据
#             cursor.executemany(insert_query, batch_data)
#         connection.commit()
#         print("Data inserted successfully")
#         cursor.close()
#         connection.close()

# 定义一个视频文件的路径
video_path = 'D:\\BaiduSyncdisk\\科研\\数据集\\animals_video\\animals.mp4'

# 对视频进行处理，提取其中的动物检测信息
video_detections = process_video(video_path)


video = cv2.VideoCapture(video_path)
# 获取视频的帧率
fps = video.get(cv2.CAP_PROP_FPS)
print("Frame rate:", fps)

# 使用前面的检测结果构建索引
index = build_index(video_detections, fps)

# 打印构建的索引以查看结果
for label, entries in index.items():
    print(f"Label: {label}")
    for entry in entries:
        print(f"  Frame: {entry['frame_index']}, Time: {entry['time_stamp']:.2f}s, Confidence: {entry['confidence']:.2f}")

csv_file_path = 'video_index.csv'
save_index_to_csv(index, csv_file_path)
print("Index has been saved to CSV file successfully.")
# insert_objects(index)
# print("Index has been saved to MySql successfully.")

# ------------------------------------------------------------------------------------------------------------------------------------------------
# #
# # 本地加载权重文件导入yolov5s的方法
# import cv2
# import torch
# import csv
# from yolov5.models.yolo import DetectionModel
# import requests
# import pandas as pd
# # import mysql.connector
# # from mysql.connector import Error
# # print(Model)
# # 使用YOLOv5目标检测算法对视频逐帧目标检测和并提取关键帧中的对象
# # 此算法基于pytorch框架，对于检测结果构建索引并存入csv文件中和MySQL数据库中
# # 优化方法：1、目标检测算法运用GPU进行加速计算，大大降低了训练耗时，提高了运算效率
# #         2、使用批处理和预编译语句优化数据库操作
#
# # 加载预训练的YOLOv5模型
# # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# # 定义本地模型权重文件的路径
# local_weights_path = 'yolov5s.pt'
# # 加载模型结构（根据您使用的具体YOLOv5版本，这里可能需要调整）
# model = DetectionModel()
# # 检查GPU是否可用并加载模型到GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)  # 修改：将模型加载到GPU
# # 加载权重
# model.load_state_dict(torch.load(local_weights_path))
# # 定义处理视频的函数
# # 导入OpenCV库，用于视频处理
# def process_video(video_path):
#     """
#     处理视频文件。
#
#     该函数尝试打开指定路径的视频文件，并进行后续处理。如果无法打开视频文件，则打印错误信息并退出函数。
#
#     参数:
#     video_path: 字符串，表示视频文件的路径。
#
#     返回值:
#     无
#     """
#     # 使用OpenCV的VideoCapture函数打开视频文件
#     cap = cv2.VideoCapture(video_path)
#     # 检查视频文件是否成功打开
#     if not cap.isOpened():
#         # 如果视频文件无法打开，则打印错误信息并返回
#         print("Error: Could not open video.")
#         return
#
#     # 初始化帧索引和检测结果列表
#     frame_index = 0
#     detections = []
#
#     # 不断读取视频帧直到结束
#     while True:
#         # 从视频捕获设备中读取一帧
#         ret, frame = cap.read()
#         # 如果帧读取失败，退出循环
#         if not ret:
#             break
#         # 将BGR格式的帧转换为RGB格式，以适应模型输入
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         # 使用模型对当前帧进行检测
#         results = model(frame_rgb)  # 模型会自动使用GPU进行推理
#         # 将检测结果转换为Pandas DataFrame格式
#         results_data = results.pandas().xyxy[0]
#         # 将当前帧的索引和检测结果添加到列表中
#         detections.append({
#             'frame_index': frame_index,
#             'detections': results_data.to_dict(orient='records')
#         })
#         # 增加帧索引
#         frame_index += 1
#
#     # 释放视频捕获设备
#     cap.release()
#     # 返回所有的检测结果
#     return detections
#
# # 构建索引（创建一个字典来存储每个对象标签及其出现的帧和时间码）
# def build_index(detections, fps):
#     """
#     构建一个索引，将检测到的对象按时间顺序索引。
#
#     参数:
#     detections: 包含多个检测结果的列表，每个检测结果包含一个帧索引和一组检测到的对象。
#     fps: 帧率，用于将帧索引转换为时间戳。
#
#     返回:
#     一个字典，键为对象名称，值为一个列表，列表中每个元素包含帧索引、时间戳和对象的置信度。
#     """
#     # 初始化一个空的索引字典
#     index = {}
#     # 遍历所有的检测结果
#     for detection in detections:
#         # 提取当前检测结果的帧索引
#         frame_index = detection['frame_index']
#         # 计算对应的时间戳
#         time_stamp = frame_index / fps  # 计算时间戳
#         # 遍历当前帧中检测到的所有对象
#         for det in detection['detections']:
#             # 提取对象的标签
#             label = det['name']
#             # 如果当前标签尚未出现在索引中，则初始化一个空的列表
#             if label not in index:
#                 index[label] = []
#             # 将当前对象的帧索引、时间戳和置信度添加到对应的列表中
#             index[label].append({'frame_index': frame_index, 'time_stamp': time_stamp, 'confidence': det['confidence']})
#     # 返回构建完成的索引字典
#     return index
#
# # 将结果保存入csv文件中
# def save_index_to_csv(index, csv_file_path):
#     """
#     将索引数据保存到CSV文件中。
#
#     参数:
#     index: 字典类型，包含标签、帧索引、时间戳和置信度等信息。
#     csv_file_path: 字符串类型，表示CSV文件的路径。
#
#     该函数打开指定的CSV文件，以写入模式创建一个writer对象，然后遍历索引数据，
#     将每条数据写入CSV文件中，每行数据包括标签、帧索引、时间戳和置信度。
#     """
#     # 以写入模式打开CSV文件，并配置不生成额外的空行
#     with open(csv_file_path, mode='w', newline='') as file:
#         # 创建CSV写入器
#         writer = csv.writer(file)
#         # 写入CSV文件的表头
#         writer.writerow(['Label', 'Frame Index', 'Time Stamp', 'Confidence'])
#
#         # 遍历索引字典中的每个标签及其对应的数据条目
#         for label, entries in index.items():
#             for entry in entries:
#                 # 将每条数据的各个字段写入CSV文件的一行
#                 # 使用f-string格式化浮点数，保留两位小数
#                 writer.writerow(
#                     [label, entry['frame_index'], f"{entry['time_stamp']:.2f}", f"{entry['confidence']:.2f}"])
#
# # 连接到MySql数据库
# # def connect_to_mysql():
# #     try:
# #         connection = mysql.connector.connect(
# #             host='localhost',
# #             user='root',
# #             password='8729512929abc',
# #             database='video_analysis'
# #         )
# #         if connection.is_connected():
# #             db_Info = connection.get_server_info()
# #             print("Connected to MySql Server version ", db_Info)
# #             return connection
# #     except Error as e:
# #         print("Error while connecting to MySql", e)
#
# # 插入检测数据到数据库
# # def insert_objects(index):
# #     connection = connect_to_mysql()
# #     if connection is not None:
# #         cursor = connection.cursor()
# #         insert_query = """
# #         INSERT INTO objects (label, frame_index, time_stamp, confidence)
# #         VALUES (%s, %s, %s, %s)
# #         """
# #         for label, entries in index.items():
# #             for entry in entries:
# #                 data_tuple = (label, entry['frame_index'], entry['time_stamp'], entry['confidence'])
# #                 cursor.execute(insert_query, data_tuple)
# #         connection.commit()
# #         print("Data inserted successfully")
# #         cursor.close()
# #         connection.close()
#
# # 改进后使用批处理和预编译语句优化数据库操作
# # （逐条执行插入操作效率低下，尤其是在处理大量数据时。所以通过使用预编译语句和批处理来优化这一过程。）
# # def insert_objects(index):
# #     connection = connect_to_mysql()
# #     if connection is not None:
# #         cursor = connection.cursor(prepared=True)
# #         insert_query = """
# #         INSERT INTO objects (label, frame_index, time_stamp, confidence)
# #         VALUES (%s, %s, %s, %s)
# #         """
# #         batch_data = []
# #         for label, entries in index.items():
# #             for entry in entries:
# #                 data_tuple = (label, entry['frame_index'], entry['time_stamp'], entry['confidence'])
# #                 batch_data.append(data_tuple)
# #                 if len(batch_data) >= 1000:  # 批量插入，每批1000条
# #                     cursor.executemany(insert_query, batch_data)
# #                     batch_data = []
# #         if batch_data:  # 插入剩余的数据
# #             cursor.executemany(insert_query, batch_data)
# #         connection.commit()
# #         print("Data inserted successfully")
# #         cursor.close()
# #         connection.close()
#
# # 定义一个视频文件的路径
# video_path = 'D:\\BaiduSyncdisk\\科研\\数据集\\animals_video\\animals.mp4'
#
# # 对视频进行处理，提取其中的动物检测信息
# video_detections = process_video(video_path)
#
# video = cv2.VideoCapture(video_path)
# # 获取视频的帧率
# fps = video.get(cv2.CAP_PROP_FPS)
# print("Frame rate:", fps)
#
# # 使用前面的检测结果构建索引
# index = build_index(video_detections, fps)
#
# # 打印构建的索引以查看结果
# for label, entries in index.items():
#     print(f"Label: {label}")
#     for entry in entries:
#         print(f"  Frame: {entry['frame_index']}, Time: {entry['time_stamp']:.2f}s, Confidence: {entry['confidence']:.2f}")
#
# csv_file_path = 'video_index.csv'
# save_index_to_csv(index, csv_file_path)
# print("Index has been saved to CSV file successfully.")
