import tkinter as tk
from tkinter import messagebox
import mysql.connector
from mysql.connector import Error
from cachetools import cached, TTLCache
import threading
import cv2
import spacy

# 基于MySQL查询性能不佳问题的改进方案
# 1、在数据库中为label字段建立索引，可以显著提高查询速度。
# CREATE INDEX idx_label ON video_analysis (label);
# 2、引入结果缓存机制：如果相同的查询被频繁执行，可以通过引入缓存机制来存储查询结果，避免重复的数据库访问。
# 3、用户界面响应性能优化(使用异步操作优化GUI响应)
# 4、对查询结果按照置信度进行排序，优先输出置信度高的视频片段内容
# 5、使用NLP技术处理复杂的文本内容，提取关键字

# 加载英语NLP模型
nlp = spacy.load("en_core_web_sm")

# 数据库连接配置
DATABASE_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '8729512929abc',
    'database': 'video_analysis'
}

# 设置缓存，最大100条记录，每条记录存活时间300秒
cache = TTLCache(maxsize=100, ttl=300)

# 同步连接到MySQL数据库
def connect_to_mysql():
    try:
        return mysql.connector.connect(**DATABASE_CONFIG)
    except Error as e:
        messagebox.showerror("Database Connection Error", f"Error connecting to MySQL: {e}")
        return None

# 提取关键词
def extract_keywords(text):
    doc = nlp(text)
    return [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]

# 实现了按照置信度排序输出，并使用提取的关键词进行搜索
@cached(cache)
def search_video_content(keywords_tuple):
    keywords = list(keywords_tuple)
    connection = connect_to_mysql()
    if connection:
        cursor = connection.cursor()
        results = []
        for keyword in keywords:
            search_query = "SELECT frame_index, time_stamp, confidence FROM objects WHERE label LIKE %s ORDER BY confidence DESC"
            cursor.execute(search_query, (f"%{keyword}%",))
            results.extend(cursor.fetchall())
        cursor.close()
        connection.close()
        return results
    return []

# 更新列表框内容
def update_listbox(results):
    listbox.delete(0, tk.END)
    if results:
        for frame_index, time_stamp, confidence in results:
            display_text = f"Frame: {frame_index}, Time: {time_stamp:.2f}s, Confidence: {confidence:.2f}"
            listbox.insert(tk.END, display_text)

# 查询按钮点击事件处理
def on_search_click():
    query = entry_label.get()
    keywords = extract_keywords(query)
    keywords_tuple = tuple(keywords)
    threading.Thread(target=perform_search, args=(keywords_tuple,)).start()

def perform_search(keywords_tuple):
    results = search_video_content(keywords_tuple)
    # 使用线程安全的方式更新 GUI
    listbox.after(0, update_listbox, results)

# 播放选定的视频片段
def play_video():
    selection = listbox.curselection()
    if not selection:
        return
    entry = listbox.get(selection[0])
    time_stamp = float(entry.split(", Time: ")[1].split('s')[0])
    video_path = 'video/animals.mp4'
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(time_stamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)

    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', frame_width, frame_height)

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
