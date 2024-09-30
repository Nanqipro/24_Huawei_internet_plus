from elasticsearch import Elasticsearch
import csv
import tkinter as tk
from tkinter import ttk
import cv2

# 基于csv文件和MySQL数据库查询性能不佳问题的改进方案--对elasticsearch的改进：
# 1、融合了GUI和elasticsearch搜索引擎来实现可视化查询和加速查询
# 2、对查询结果按照置信度进行排序，优先输出置信度高的视频片段内容，提高准确性
# 3、采用分页输出，用于提高查询性能和响应速度，降低内存消耗，改善用户体验，减少服务器负载等

# 连接到本地Elasticsearch服务
es = Elasticsearch("http://localhost:9200")

# 创建Elasticsearch索引
def create_index():
    try:
        es.indices.create(index='video_index', ignore=400, body={
            "settings": {
                "analysis": {
                    "analyzer": {
                        "default": {
                            "type": "standard",
                            "stopwords": "_english_"
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "label": {"type": "text"},
                    "frame_index": {"type": "integer"},
                    "time_stamp": {"type": "float"},
                    "confidence": {"type": "float"}
                }
            }
        })
    except Exception as e:
        print(f"创建索引时出错: {e}")

# 从CSV文件读取数据并索引到Elasticsearch
def index_data_from_csv(csv_file_path):
    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                doc = {
                    "label": row['Label'],
                    "frame_index": int(row['Frame Index']),
                    "time_stamp": float(row['Time Stamp']),
                    "confidence": float(row['Confidence'])
                }
                es.index(index='video_index', document=doc)
            except Exception as e:
                print(f"索引数据时出错: {e}")

# 按照置信度从大到小排序，优先输出高置信度的视频片段内容
def search_videos(page=1, size=10):
    query = entry.get()
    from_record = (page - 1) * size
    body = {
        "query": {
            "match": {
                "label": query
            }
        },
        "sort": [  # 添加排序规则
            {
                "confidence": {
                    "order": "desc"
                }
            }
        ],
        "from": from_record,
        "size": size
    }
    results = es.search(index='video_index', body=body)
    listbox.delete(0, tk.END)
    for hit in results['hits']['hits']:
        listbox.insert(tk.END,
                       f"Label: {hit['_source']['label']}, "
                       f"Frame Index: {hit['_source']['frame_index']}, "
                       f"Time Stamp: {hit['_source']['time_stamp']:.2f}, "
                       f"Confidence: {hit['_source']['confidence']:.2f}")

    total_results = results['hits']['total']['value']
    update_pagination(page, size, total_results)


# 更新分页按钮状态
def update_pagination(page, size, total):
    current_page_label.config(text=f"Page {page} of {max((total + size - 1) // size, 1)}")
    prev_button.config(state=tk.NORMAL if page > 1 else tk.DISABLED)
    next_button.config(state=tk.NORMAL if (page - 1) * size + size < total else tk.DISABLED)

# 分页控制
def next_page():
    current_page = int(current_page_label.cget("text").split()[1])
    search_videos(page=current_page + 1)

def prev_page():
    current_page = int(current_page_label.cget("text").split()[1])
    search_videos(page=current_page - 1)

# 查询到静态视频片段
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

# 查询并跳转到可播放的视频
def play_video():
    selection = listbox.curselection()
    if not selection:
        return
    entry_text = listbox.get(selection[0])
    time_stamp = float(entry_text.split(', Time Stamp: ')[1].split(',')[0])
    cap = cv2.VideoCapture('video/animals.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(float(time_stamp) * fps)
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
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# 设置GUI
root = tk.Tk()
root.title("Video Search Engine")

entry = tk.Entry(root, width=70)
entry.pack()

search_button = tk.Button(root, text="Search", command=lambda: search_videos(page=1))
search_button.pack()

listbox = tk.Listbox(root, width=70, height=10)
listbox.pack()

play_button = tk.Button(root, text="Play Video", command=play_video)
play_button.pack()

prev_button = tk.Button(root, text="Previous", command=prev_page)
prev_button.pack(side=tk.LEFT, padx=(20, 10))

current_page_label = tk.Label(root, text="Page 1 of 1")
current_page_label.pack(side=tk.LEFT)

next_button = tk.Button(root, text="Next", command=next_page)
next_button.pack(side=tk.LEFT, padx=(10, 20))

# 初始化Elasticsearch索引并索引数据
create_index()
index_data_from_csv('video_index.csv')

root.mainloop()

