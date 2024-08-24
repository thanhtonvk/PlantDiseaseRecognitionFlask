from utils import onnx_model_inference
import cv2
import numpy as np


class NhanDien:

    def __init__(self, model_type):
        self.model_type = model_type
        if self.model_type == 0:
            self.model = onnx_model_inference('models/nhandiencaydau.onnx')
        elif self.model_type == 1:
            self.model = onnx_model_inference('models/nhandiencayngo.onnx')

    def preprocess(self, image: np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = (image-127.5)/127.5
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return image

    def predict(self, image: np.ndarray):
        image = self.preprocess(image).astype('float32')
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: image})[0].flatten()
        outputs = [int(out*100) for out in outputs]
        print(outputs)
        return outputs
