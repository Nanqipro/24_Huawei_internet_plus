from ultralytics import YOLO
# 训练
# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("/app/ZZQ/zj/yolov8_animals/yolov8_init/ultralytics-main/ultralytics/runs/detect/train/weights/best.pt")  # load a pretrained model (recommended for training)

# # 验证
# # Use the model
# # model.train(data="/app/ZZQ/zj/yolov8_face/ultralytics-main/ultralytics/dataset/Faces.v2-v2.yolov8/data.yaml", epochs=300)
# metrics = model.val(data="/app/ZZQ/zj/yolov8_animals/yolov8_init/ultralytics-main/ultralytics/dataset/data.yaml")  # evaluate model performance on the validation set
# print(metrics)

# 测试
results = model("/app/ZZQ/zj/yolov8_animals/yolov8_init/ultralytics-main/ultralytics/dataset/test_temp")  # predict on an image
for result in results:
    result.save()
# # path = model.export(format="onnx")  # export the model to ONNX format

# 验证是否有可用的GPU
import torch
print(torch.cuda.is_available())
