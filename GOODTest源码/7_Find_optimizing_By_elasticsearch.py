from elasticsearch import Elasticsearch
import csv

# 针对使用MySQL查询性能不佳的问题这里引入elasticsearch搜索引擎：
# Elasticsearch 是一个分布式搜索引擎,采用倒排索引机制，引入elasticsearch搜索引擎查询速率显著提升，对于大规模查询效果显著

es = Elasticsearch("http://localhost:9200")
es = es.options(ignore_status=[400])
if not es.indices.exists(index='video_index'):
    es.indices.create(index='video_index')

    # 定义映射
    mapping = {
        "properties": {
            "label": {"type": "text"},
            "frame_index": {"type": "integer"},
            "time_stamp": {"type": "float"},
            "confidence": {"type": "float"}
        }
    }

    # 应用映射
    es.indices.put_mapping(index='video_index', body=mapping)

# 索引数据到Elasticsearch
def index_data_to_elasticsearch(csv_file_path):
    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            doc = {
                "label": row['Label'],
                "frame_index": int(row['Frame Index']),
                "time_stamp": float(row['Time Stamp']),
                "confidence": float(row['Confidence'])
            }
            es.index(index='video_index', document=doc)
    es.indices.refresh(index='video_index')

index_data_to_elasticsearch('video_index.csv')

# 搜索数据
def search_video_content(query):
    body = {
        "query": {
            "match": {
                "label": query
            }
        },
        "size": 100
    }
    result = es.search(index='video_index', body=body)
    for hit in result['hits']['hits']:
        print(hit['_source'])

label_to_search = input("Enter the label to search for: ")
search_video_content(label_to_search)
