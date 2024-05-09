import copy
import time
from abc import abstractmethod
from queue import Empty
import cv2
import numpy as np
import torch
from PIL import ImageFont, Image, ImageDraw
from PyQt5.QtGui import QPixmap, QImage
from pipeline_module.core.base_module import BaseModule, TASK_DATA_OK, DictData
from utils.vis import draw_keypoints136
box_color = (0, 255, 0)
cheating_box_color = (0, 0, 255)
draw_keypoints_default = False
def draw_frame(data, draw_keypoints=draw_keypoints_default, fps=-1):
    frame = data.frame.copy()
    pred = data.detections
    preds_kps = data.keypoints
    preds_scores = data.keypoints_scores
    if pred.shape[0] > 0:
        if draw_keypoints and preds_kps is not None:
            draw_keypoints136(frame, preds_kps, preds_scores)
        frame_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame_pil)
        for det, class_prob, best_pred in zip(pred, data.classes_probs, data.best_preds):
            det = det.to(torch.int)
            class_name = data.classes_names[best_pred]
            show_text = f"{class_name}"
            show_color = box_color if best_pred == 0 else cheating_box_color
            draw.rectangle((det[0], det[1], det[2], det[3]), outline=show_color, width=2)
            fontText = ImageFont.truetype("resource/font/NotoSansCJKkr-Black.otf",
                                          int(40 * (min(det[2] - det[0], det[3] - det[1])) / 200),
                                          encoding="utf-8")
            draw.text((det[0], det[1]), show_text, show_color, font=fontText)
        frame = np.asarray(frame_pil)
        for (r, t) in data.head_pose:
            data.draw_axis(frame, r, t)
    frame=frame.copy()
    cv2.putText(frame, "FPS: %d" % fps, (0, 52), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
    data.frame_anno = frame
class DrawModule(BaseModule):
    def __init__(self):
        super(DrawModule, self).__init__()
        self.last_time = time.time()
    def process_data(self, data):
        current_time = time.time()
        fps = 1 / (current_time - self.last_time)
        frame = data.frame
        pred = data.detections
        preds_kps = data.keypoints
        preds_scores = data.keypoints_scores
        for det in pred:
            show_text = "person: %.2f" % det[4]
            cv2.rectangle(frame, (det[0], det[1]), (det[2], det[3]), box_color, 2)
            cv2.putText(frame, show_text,
                        (det[0], det[1]),
                        cv2.FONT_HERSHEY_COMPLEX,
                        float((det[2] - det[0]) / 200),
                        box_color)
        if preds_kps is not None:
            draw_keypoints136(frame, preds_kps, preds_scores)
        cv2.putText(frame, "FPS: %d" % fps, (0, 52), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
        cv2.imshow("yolov8", frame)
        cv2.waitKey(40)
        self.last_time = current_time
        return TASK_DATA_OK
    def open(self):
        super(DrawModule, self).open()
        pass
class FrameDataSaveModule(BaseModule):
    def __init__(self, app):
        super(FrameDataSaveModule, self).__init__()
        self.last_time = time.time()
        self.app = app
    def process_data(self, data):
        current_time = time.time()
        fps = 1 / (current_time - self.last_time)
        frame = data.frame
        pred = data.detections
        preds_kps = data.keypoints
        preds_scores = data.keypoints_scores
        for det in pred:
            show_text = "person: %.2f" % det[4]
            cv2.rectangle(frame, (det[0], det[1]), (det[2], det[3]), box_color, 2)
            cv2.putText(frame, show_text,
                        (det[0], det[1]),
                        cv2.FONT_HERSHEY_COMPLEX,
                        float((det[2] - det[0]) / 200),
                        box_color)
        if preds_kps is not None:
            draw_keypoints136(frame, preds_kps, preds_scores)
        cv2.putText(frame, "FPS: %d" % fps, (0, 52), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
        self.app.video_screen.setPixmap(self.cvImg2qtPixmap(frame))
        self.last_time = current_time
        return TASK_DATA_OK
    @staticmethod
    def cvImg2qtPixmap(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_height, image_width, image_depth = frame.shape
        frame = QImage(frame.data, image_width, image_height,
                       image_width * image_depth,
                       QImage.Format_RGB888)
        return QPixmap.fromImage(frame)
    def open(self):
        super(FrameDataSaveModule, self).open()
        pass
class DataDealerModule(BaseModule):
    def __init__(self, push_frame_func, interval=0.06, skippable=False):
        super(DataDealerModule, self).__init__(skippable=skippable)
        self.last_time = time.time()
        self.push_frame_func = push_frame_func
        self.last_data = None
        self.interval = interval
        self.size_waiting = True
        self.queue_threshold = 10
    @abstractmethod
    def deal_skipped_data(self, data: DictData, last_data: DictData) -> DictData:
        pass
    @abstractmethod
    def draw_frame(self, data, fps):
        pass
    def process_data(self, data):
        if hasattr(data, 'skipped') and self.last_data is not None:
            data = self.deal_skipped_data(data, copy.copy(self.last_data))
        else:
            self.last_data = data
        current_time = time.time()
        interval = (current_time - self.last_time)
        fps = 1 / interval
        data.fps = fps
        self.draw_frame(data, fps=fps)
        data.interval = interval
        self.last_time = current_time
        self.push_frame_func(data)
        if hasattr(data, 'source_fps'):
            time.sleep(1 / data.source_fps * (1 + self.self_balance_factor()))
        else:
            time.sleep(self.interval)
        return TASK_DATA_OK
    def self_balance_factor(self):
        factor = max(-0.999, (self.queue.qsize() / 20 - 0.5) / -0.5)
        return factor
    def product_task_data(self):
        if self.queue.qsize() == 0:
            self.size_waiting = True
        if self.queue.qsize() > self.queue_threshold or not self.size_waiting:
            self.size_waiting = False
            try:
                task_data = self.queue.get(block=True, timeout=1)
                return task_data
            except Empty:
                return self.ignore_task_data
        else:
            time.sleep(1)
            return self.ignore_task_data
    def put_task_data(self, task_data):
        self.queue.put(task_data)
    def open(self):
        super(DataDealerModule, self).open()
        pass
class CheatingDetectionVisModule(DataDealerModule):
    def __init__(self, push_frame_func, interval=0.06, skippable=False):
        super(CheatingDetectionVisModule, self).__init__(push_frame_func, interval, skippable)
    def deal_skipped_data(self, data: DictData, last_data: DictData) -> DictData:
        frame = data.frame
        data = last_data
        data.skipped = None
        data.frame = frame
        data.detections = data.detections.clone()
        if data.detections.numel()>0:
            data.detections[:, :4] += torch.rand_like(data.detections[:, :4]) * 3
        return data
    def draw_frame(self, data, fps):
        draw_frame(data, fps=fps)

