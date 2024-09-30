from ultralytics import YOLO

# 加载模型
model = YOLO("runs/detect/train10/weights/best.pt")  # 加载预训练模型（建议用于训练）

# 使用模型
# metrics = model.val(data="datasets/data/data.yaml")  # 在验证集上评估模型性能
# print(metrics)
results = model("datasets/data/test/images")
for result in results:
    result.save()
