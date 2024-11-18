import numpy as np
from ultralytics import YOLO
names = {0: 'Sọc tán lá giai đoạn cao', 1: 'Sọc tán lá giai đoạn thấp', 2: 'Sọc tán lá giai đoạn trung bình', 3: 'Đốm lá giai đoạn cao', 4: 'Đốm lá giai đoạn thấp', 5: 'lá phát hiện giai đoạn trung bình'}
model = YOLO("models/nhan_dien_benh_la_cam.pt")

def predictCayQue(image: np.ndarray):
    result = model.predict(image, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().detach().numpy().astype("int")
    cls = result.boxes.cls.cpu().detach().numpy().astype("int")
    classes = [names[i] for i in cls]
    scores = result.boxes.conf.cpu().detach().numpy().astype('float')
    return boxes, classes, scores

