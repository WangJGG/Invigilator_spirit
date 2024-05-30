import numpy as np
import torch
from ultralytics import YOLO
class YoloV8Detector:
    def __init__(self, weights, device):
        self.model = YOLO(weights)
        self.device = device
    def detect(self, img):
        result = self.model(img, verbose=False)
        all_boxes = []
        for r in result:
            for box in r.boxes:
                if int(box.cls[0]) == 0:
                    pred=box.data.cpu().numpy()
                    all_boxes.extend(pred)
        all_boxes_tensor = torch.from_numpy(np.array(all_boxes))#.to(self.device)
        return all_boxes_tensor
