import tkinter as tk
from tkinter import messagebox
import mysql.connector
from mysql.connector import Error
import cv2
import spacy

# 生成图形化用户界面，使用关系型数据库MySQL来处理查询请求并播放视频片段
# 相比于简单的用户查询和返回，这里做了一些改进：
# 1、使用NLP技术处理复杂的文本内容，提取关键字
# 2、能够定位到对应的视频片段并播放视频，也能查询静态图片
# 3、对查询结果按照置信度由大到小进行了排序，置信度高的查询结果更加精确
# 4、将视频播放窗口减小为原来的一半，以获得更好的用户体验


# 加载英语NLP模型
nlp = spacy.load("en_core_web_sm")

# 数据库连接配置
DATABASE_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '8729512929abc',
    'database': 'video_analysis'
}

# 提取关键字
def extract_keywords(text):
    doc = nlp(text)
    keywords = [token.text.lower() for token in doc if token.pos_ in ['NOUN', 'PROPN']]
    return keywords

# 连接到MySQL数据库
def connect_to_mysql():
    try:
        return mysql.connector.connect(**DATABASE_CONFIG)
    except Error as e:
        messagebox.showerror("Database Connection Error", f"Error connecting to MySQL: {e}")
        return None

# 从数据库中查询视频内容并排序结果
def search_video_content(keyword):
    connection = connect_to_mysql()
    if connection:
        cursor = connection.cursor()
        # 添加ORDER BY子句按confidence字段降序排序
        search_query = "SELECT frame_index, time_stamp, confidence FROM objects WHERE label LIKE %s ORDER BY confidence DESC"
        cursor.execute(search_query, (f"%{keyword}%",))
        results = cursor.fetchall()
        cursor.close()
        connection.close()
        return results
    return []

# 查询按钮点击事件处理
def on_search_click():
    query = entry_label.get()
    keywords = extract_keywords(query)
    results = []
    for keyword in keywords:
        results.extend(search_video_content(keyword))
    update_listbox(results)

# 更新列表框内容
def update_listbox(results):
    listbox.delete(0, tk.END)
    if results:
        for frame_index, time_stamp, confidence in results:
            display_text = f"Frame: {frame_index}, Time: {time_stamp:.2f}s, Confidence: {confidence:.2f}"
            listbox.insert(tk.END, display_text + f"||{time_stamp}")  # 存储时间戳于字符串末尾以备后用


# 生成静态图
def play_video_static():
    selection = listbox.curselection()
    if not selection:
        return
    _, time_stamp = listbox.get(selection[0])
    cap = cv2.VideoCapture('video/animals.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(float(time_stamp) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Video', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    cap.release()

# 播放视频片段
def play_video():
    selection = listbox.curselection()
    if not selection:
        return
    entry = listbox.get(selection[0])
    time_stamp = float(entry.split("||")[-1])
    video_path = 'video/animals.mp4'
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(time_stamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # 获取视频帧的宽度和高度
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)

    # 设置窗口名称和大小
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', frame_width, frame_height)

    # 开始播放视频
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):  # 按 'q' 键退出播放
            break
    cap.release()
    cv2.destroyAllWindows()


# 创建GUI
root = tk.Tk()
root.title("Video Content Search")

entry_label = tk.Entry(root)
entry_label.pack(pady=10)

search_button = tk.Button(root, text="Search", command=on_search_click)
search_button.pack(pady=5)

listbox = tk.Listbox(root, width=60, height=10)
listbox.pack(pady=10)

play_button = tk.Button(root, text="Play Video", command=play_video)
play_button.pack(pady=5)

root.mainloop()