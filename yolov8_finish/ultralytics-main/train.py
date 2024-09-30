from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("/home/nanqipro01/yolov8/yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="/home/nanqipro01/yolov8/ultralytics-main/data/demo01.v1i.yolov8/data.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format