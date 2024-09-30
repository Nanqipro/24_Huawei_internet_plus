import cv2
import random
import numpy as np
import onnxruntime as ort
import our_tool as our_tool
 
# names = ["face", "phone"]
 
class ONNX_engine():

    def __init__(self, weights_path, size, cuda = False) -> None:
        self.names = ['face', 'phone', 'openeye', 'closedeye', 'yawn', 'noyawn']# 类别
        self.device = cuda# 设备
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device else ['CPUExecutionProvider']
        # 加载推理模型，创建 onnxruntime InferenceSession对象
        self.session = ort.InferenceSession(weights_path, providers=providers)
        self.img_new_shape = (size, size)# size
        # 检测框颜色
        self.colors = {name: [random.randint(0, 255) for _ in range(3)] for i, name in enumerate(self.names)}

        
    # 推理
    def predict(self, image):
        input_name = [i.name for i in self.session.get_inputs()] # 获取 input node name
        #input_name:['images']
        output_name = [i.name for i in self.session.get_outputs()] # 获取 output node name
        #output_name:['output']
        # 推理
        outputs = self.session.run(output_name, {input_name[0]: image})[0]
        # print(type(outputs))
        # print(outputs.shape)
        # print(outputs)
        return outputs
        #outputs:array([],shape=(0,7),dtype=float32)

    # 前后处理
    def preprocess(self, frame):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1] 
        frame_area = frame_height * frame_width
        temp = []
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = img.copy()
        im, ratio, dwdh = self.letterbox(image, auto=False)
        #dwdh:(0.0, 50.0)
        # 推理
        # t1 = time.time()
        outputs = self.predict(im)
        # print("推理时间", (time.time() - t1) * 1000, ' ms')

        ori_images = [img.copy()]
        # print(outputs)
        
        for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(outputs):
            image = ori_images[int(batch_id)]
            box = np.array([x0, y0, x1, y1])
            box -= np.array(dwdh * 2)
            box /= ratio
            xyxy = box.round().astype(np.int32).tolist()
            xywh = our_tool.xyxy2xywh(xyxy)
            temp.append([int(cls_id), xywh])
        
        result = our_tool.func_clean(temp, frame_area, frame_height, frame_width)
        return result
    
 
    def letterbox(self, im, color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # 调整大小和垫图像，同时满足跨步多约束
        shape = im.shape[:2]  # current shape [height, width]
        new_shape = self.img_new_shape

        # 如果是单个size的话，就在这里变成一双
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # 尺度比 (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # 只缩小，不扩大(为了更好的val mAP)
            r = min(r, 1.0)

        # 计算填充
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # 最小矩形区域
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        im = im.transpose((2, 0, 1)) #将图像的维度顺序从(H, W, C)变为(C, H, W)，其中H是高度，W是宽度，C是通道数。
        im = np.expand_dims(im, 0)
        im = np.ascontiguousarray(im)
        im = im.astype(np.float32)
        im /= 255
        return im, r, (dw, dh)
    
    
