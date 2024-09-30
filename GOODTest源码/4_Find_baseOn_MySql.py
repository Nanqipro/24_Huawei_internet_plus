import tkinter as tk
from tkinter import scrolledtext
import mysql.connector
from mysql.connector import Error

# 链接MySql数据库并且在数据库中查询视频片段,使用了GUI来设计了一个简单的查询窗口
def connect_to_mysql():
    try:
        return mysql.connector.connect(
            host='localhost',
            user='root',
            password='8729512929abc',
            database='video_analysis'
        )
    except Error as e:
        print("Error while connecting to MySql", e)
        return None

def search_video_content(label_to_search):
    connection = connect_to_mysql()
    if connection is not None:
        cursor = connection.cursor()
        search_query = "SELECT frame_index, time_stamp, confidence FROM objects WHERE label = %s"
        cursor.execute(search_query, (label_to_search,))
        results = cursor.fetchall()
        cursor.close()
        connection.close()
        return results

def on_search_click():
    label_to_search = entry_label.get()
    results = search_video_content(label_to_search)
    text_area.config(state=tk.NORMAL)
    text_area.delete(1.0, tk.END)
    if results:
        result_text = f"Found {len(results)} results for label '{label_to_search}':\n"
        for result in results:
            result_text += f"Frame Index: {result[0]}, Time Stamp: {result[1]:.2f}s, Confidence: {result[2]:.2f}\n"
        text_area.insert(tk.END, result_text)
    else:
        text_area.insert(tk.END, "No results found.")
    text_area.config(state=tk.DISABLED)

root = tk.Tk()
root.title("Video Content Search")

label_prompt = tk.Label(root, text="Enter the label to search for:")
entry_label = tk.Entry(root)
search_button = tk.Button(root, text="Search", command=on_search_click)
text_area = scrolledtext.ScrolledText(root, width=40, height=10, state=tk.DISABLED)

label_prompt.pack(pady=10)
entry_label.pack(pady=10)
search_button.pack(pady=10)
text_area.pack(padx=10, pady=10)

root.mainloop()

