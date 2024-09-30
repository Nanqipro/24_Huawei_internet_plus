import spacy
import csv

# 基于自然语言处理提取关键字来获取文本标签的视频内容索引,
# 能够在复杂的文本中提取关键字，并且精确定位到以关键字为Label的视频片段信息

# 加载英语NLP模型
nlp = spacy.load("en_core_web_sm")

# 提取关键字
def extract_keywords(text):
    # 使用spaCy处理文本
    doc = nlp(text)
    keywords = [token.text.lower() for token in doc if token.pos_ in ['NOUN', 'PROPN']]
    return keywords

#根据关键字进行查询
def search_video_content(keywords, csv_file_path):
    results = []
    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if any(keyword in row['Label'].lower() for keyword in keywords):
                results.append((row['Label'], row['Frame Index'], row['Time Stamp'], row['Confidence']))
    return results

query = input("Enter your search query: ")
keywords = extract_keywords(query)

csv_file_path = 'video_index.csv'
search_results = search_video_content(keywords, csv_file_path)

if search_results:
    print(f"Found {len(search_results)} results:")
    for result in search_results:
        print(f"Label: {result[0]}, Frame Index: {result[1]}, Time Stamp: {result[2]}s, Confidence: {result[3]}")
else:
    print("No results found.")
