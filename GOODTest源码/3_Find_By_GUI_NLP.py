import tkinter as tk
from tkinter import ttk, messagebox
import csv
import cv2
import spacy

# 在前一个技术的基础上，引入GUI图形化界面,实现了一个简单的搜索窗口
# 使用NLP模型对输入的文本提取关键字，提高索引准确性，使得能够应对复杂的文本
# 可定位相关视频片段（静态）

# 加载英语NLP模型
nlp = spacy.load("en_core_web_sm")

# 关键字提取
def extract_keywords(text):
    doc = nlp(text)
    keywords = [token.text.lower() for token in doc if token.pos_ in ['NOUN', 'PROPN']]
    return keywords

# 根据关键字搜索视频片段内容
def search_video_content(keywords, csv_file_path):
    results = []
    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if any(keyword in row['Label'].lower() for keyword in keywords):
                results.append((row['Label'], row['Frame Index'], row['Time Stamp'], row['Confidence']))
    return results

# 根据时间戳定位相关视频片段
def open_video_at_time(video_path, time_stamp):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(float(time_stamp) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Video', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    cap.release()

def perform_search():
    query = entry.get()
    keywords = extract_keywords(query)
    results = search_video_content(keywords, 'video_index.csv')
    listbox.delete(0, tk.END)
    for result in results:
        listbox.insert(tk.END, result)

def on_listbox_select(event):
    widget = event.widget
    index = int(widget.curselection()[0])
    value = widget.get(index)
    video_path = 'video/animals.mp4'
    open_video_at_time(video_path, value[2])

root = tk.Tk()
root.title("Video Content Search")

label = tk.Label(root, text="Enter your search query:")
label.pack()

entry = tk.Entry(root)
entry.pack()

search_button = tk.Button(root, text="Search", command=perform_search)
search_button.pack()

listbox = tk.Listbox(root, width=50, height=10)
listbox.pack()
listbox.bind('<<ListboxSelect>>', on_listbox_select)

root.mainloop()
