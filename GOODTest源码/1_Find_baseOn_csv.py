import csv
# 基于关键字的视频内容索引，查询结果定位到相应的视频静态片段,在video_index.csv文件中查询
def search_video_content(label_to_search, csv_file_path):
    results = []
    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Label'].lower() == label_to_search.lower():
                results.append((row['Frame Index'], row['Time Stamp'], row['Confidence']))

    return results

label_to_search = input("Enter the label to search for: ")

csv_file_path = 'video_index.csv'

search_results = search_video_content(label_to_search, csv_file_path)

if search_results:
    print(f"Found {len(search_results)} results for label '{label_to_search}':")
    for result in search_results:
        print(f"Frame Index: {result[0]}, Time Stamp: {result[1]}s, Confidence: {result[2]}")
else:
    print(f"No results found for label '{label_to_search}'.")
