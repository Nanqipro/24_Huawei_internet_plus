from ultralytics import YOLO
import torch
# 显式地将模型移动到 GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO(r"C:\Users\nanqipro\Desktop\yolov8_init\yolov8n.pt" )  # load a pretrained model (recommended for training)
model.to(device)
# Use the model
model.train(data=r"C:\Users\nanqipro\Desktop\yolov8_init\ultralytics-main\ultralytics\dataset\data.yaml", epochs=300,device=device)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format


