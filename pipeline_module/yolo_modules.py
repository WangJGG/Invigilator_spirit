from models.yolo_detector import YoloV5Detector
from pipeline_module.core.base_module import BaseModule, TASK_DATA_OK
class YoloV8Module(BaseModule):
    def __init__(self, weights, device, skippable=True):
        super(YoloV8Module, self).__init__(skippable=skippable)
        self.weights = weights
        self.detector = YoloV5Detector(self.weights, device)
    def process_data(self, data):
        data.detections = self.detector.detect(data.frame)
        return TASK_DATA_OK
    def open(self):
        super(YoloV8Module, self).open()
        pass
