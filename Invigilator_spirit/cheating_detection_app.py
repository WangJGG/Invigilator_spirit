import csv
import os
import time
from itertools import islice
from threading import Thread, Lock
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QWidget
import matplotlib
matplotlib.use('Agg')
from pipeline_module.core.base_module import DictData
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from pipeline_module.classroom_action_module import CheatingActionModule
from pipeline_module.core.task_solution import TaskSolution
from pipeline_module.pose_modules import AlphaPoseModule
from pipeline_module.video_modules import VideoModule
from pipeline_module.vis_modules import CheatingDetectionVisModule
from pipeline_module.yolo_modules import YoloV5Module
from Invigilator_spirit.list_items import VideoSourceItem, RealTimeCatchItem, FrameData
from ui.cheating_detection import Ui_CheatingDetection
from utils.common import second2str, OffsetList
yolov5_weight = './weights/Yolov8s_Classroom.pt'#yolov8s.pt 原模型文件
alphapose_weight = './weights/Alphapose_halpe136_mobile.torchscript.pth'
classroom_action_weight = './weights/classroom_action_classifier.torchscript.pth'
device = 'cuda'
class CheatingDetectionApp(QWidget, Ui_CheatingDetection):
    add_cheating_list_signal = QtCore.pyqtSignal(DictData)
    push_frame_signal = QtCore.pyqtSignal(DictData)
    def __init__(self, parent=None):
        super(CheatingDetectionApp, self).__init__(parent)
        self.setupUi(self)
        self.video_source = 0
        self.frame_data_list = OffsetList()
        self.opened_source = None
        self.playing = None
        self.playing_real_time = False
        self.num_of_passing = 0
        self.num_of_peep = 0
        self.num_of_gazing_around = 0
        self.open_source_lock = Lock()
        self.video_resource_list.itemClicked.connect(lambda item: self.open_source(item.src))
        self.video_resource_file_list.itemClicked.connect(lambda item: self.open_source(item.src))
        self.close_source_btn.clicked.connect(self.close_source)
        self.play_video_btn.clicked.connect(self.play_video)
        self.stop_playing_btn.clicked.connect(self.stop_playing)
        self.video_process_bar.valueChanged.connect(self.change_frame)
        self.push_frame_signal.connect(self.push_frame)
        # 其他事件
        def local_to_cheater(x):
            self.stop_playing()
            self.video_process_bar.setValue(x.frame_num)
        self.cheating_list.itemClicked.connect(local_to_cheater)
        self.real_time_catch_list.itemClicked.connect(local_to_cheater)
        self.add_cheating_list_signal.connect(self.add_cheating_list)
        self.init_video_source()
        self.init_cheating_img_data()
    def init_cheating_img_data(self):
        self.cheating_list_time = []
        self.cheating_list_count_data = dict(
            传纸条=[],
            低头偷看=[],
            东张西望=[]
        )
    def init_video_source(self):
        VideoSourceItem(self.video_resource_list, "摄像头", 0).add_item()
        local_source = 'resource/videos/cheating_detection'
        if not os.path.exists(local_source):
            os.makedirs(local_source)
        else:
            print(f"本地视频目录已创建: {local_source}")
        videos = [*filter(lambda x: x.endswith('.mp4'), os.listdir(local_source))]
        for video_name in videos:
            VideoSourceItem(self.video_resource_file_list,
                            video_name,
                            os.path.join(local_source, video_name),
                            ico_src=':/videos/multimedia.ico').add_item()
        with open('resource/video_sources.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in islice(reader, 1, None):
                VideoSourceItem(self.video_resource_list, row[0], row[1],
                                ico_src=':/videos/webcam.ico').add_item()
    def cheating_list_filter(self, idx):
        if idx == 0:
            for i in range(self.cheating_list.count()):
                self.cheating_list.item(i).setHidden(False)
        elif idx == 1:
            for i in range(self.cheating_list.count()):
                item = self.cheating_list.item(i)
                item.setHidden(item.data_.num_of_passing == 0)
        elif idx == 2:
            for i in range(self.cheating_list.count()):
                item = self.cheating_list.item(i)
                item.setHidden(item.data_.num_of_peep == 0)
        elif idx == 3:
            for i in range(self.cheating_list.count()):
                item = self.cheating_list.item(i)
                item.setHidden(item.data_.num_of_gazing_around == 0)
    def open_source(self, source):
        self.open_source_lock.acquire(blocking=True)
        if self.opened_source is not None:
            self.close_source()
        frame = np.zeros((720, 960, 3), np.uint8)
        (f_w, f_h), _ = cv2.getTextSize("Loading", cv2.FONT_HERSHEY_TRIPLEX, 1, 2)
        cv2.putText(frame, "Loading", (int((960 - f_w) / 2), int((720 - f_h) / 2)),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    1, (255, 255, 255), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.video_screen.width() - 9, self.video_screen.height() - 9))  # 调整图像大小
        image_height, image_width, image_depth = frame.shape
        frame = QImage(frame.data, image_width, image_height,
                       image_width * image_depth,
                       QImage.Format_RGB888)
        self.video_screen.setPixmap(QPixmap.fromImage(frame))
        def open_source_func(self):
            fps = 25
            self.opened_source = TaskSolution() \
                .set_source_module(VideoModule(source, fps=fps)) \
                .set_next_module(YoloV5Module(yolov5_weight, device)) \
                .set_next_module(AlphaPoseModule(alphapose_weight, device)) \
                .set_next_module(CheatingActionModule(classroom_action_weight)) \
                .set_next_module(CheatingDetectionVisModule(lambda d: self.push_frame_signal.emit(d)))
            self.opened_source.start()
            self.playing_real_time = True
            self.open_source_lock.release()
        Thread(target=open_source_func, args=[self]).start()
    def init_sceeen(self):
        frame = np.zeros((720, 960, 3), np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.video_screen.width() - 9, self.video_screen.height() - 9))  # 调整图像大小
        image_height, image_width, image_depth = frame.shape
        frame = QImage(frame.data, image_width, image_height,  # 创建QImage格式的图像，并读入图像信息
                       image_width * image_depth,
                       QImage.Format_RGB888)
        self.video_screen.setPixmap(QPixmap.fromImage(frame))
        self.time_process_label.setText("00:00:00/00:00:00")
    def close_source(self):
        if self.opened_source is not None:
            self.stop_playing()
            self.opened_source.close()
            self.opened_source = None
            self.frame_data_list.clear()
            self.video_process_bar.setMaximum(-1)
            self.playing_real_time = False
            self.cheating_list.clear()
            self.real_time_catch_list.clear()
            self.init_cheating_img_data()
            self.init_sceeen()
    def push_frame(self, data):
        try:
            max_index = self.frame_data_list.max_index()
            time_process = self.frame_data_list[max_index].time_process if len(self.frame_data_list) > 0 else 0
            data.time_process = time_process + data.interval
            self.frame_data_list.append(data)
            while len(self.frame_data_list) > 5000:
                self.frame_data_list.pop()
            self.video_process_bar.setMinimum(self.frame_data_list.min_index())
            self.video_process_bar.setMaximum(self.frame_data_list.max_index())
            data.frame_num = max_index + 1
            if data.num_of_cheating > 0 and self.check_cheating_change(data):
                self.add_cheating_list_signal.emit(data)
            if self.playing_real_time:
                self.video_process_bar.setValue(self.video_process_bar.maximum())
        except Exception as e:
            print(e)
    def check_cheating_change(self, data):
        cond = all([self.num_of_passing >= data.num_of_passing,
                    self.num_of_peep >= data.num_of_peep,
                    self.num_of_gazing_around >= data.num_of_gazing_around])
        self.num_of_passing = data.num_of_passing
        self.num_of_peep = data.num_of_peep
        self.num_of_gazing_around = data.num_of_gazing_around
        return not cond
    def playing_video(self):
        try:
            while self.playing is not None and not self.playing_real_time:
                current_frame = self.video_process_bar.value()
                max_frame = self.video_process_bar.maximum()
                data = self.frame_data_list[current_frame]
                if current_frame < 0:
                    continue
                elif current_frame < max_frame:
                    if current_frame < max_frame:
                        self.video_process_bar.setValue(current_frame + 1)
                    time.sleep(data.interval)
                else:
                    self.stop_playing()
                    self.playing_real_time = True
        except Exception as e:
            print(e)

    def add_cheating_list(self, data):
        try:
            FrameData(self.cheating_list, data).add_item()
            while self.cheating_list.count() > self.cheating_list_spin.value():
                self.cheating_list.takeItem(0)
            frame = data.frame
            detections = data.detections
            cheating_types = data.pred_class_names
            time_process = data.time_process
            frame_num = data.frame_num
            best_preds = data.best_preds
            for detection, cheating_type, best_pred in zip(detections, cheating_types, best_preds):
                if best_pred == 0:
                    continue
                detection = detection[:4].clone()
                detection[2:] = detection[2:] - detection[:2]
                RealTimeCatchItem(self.real_time_catch_list, frame, detection, time_process, cheating_type,
                                  frame_num).add_item()
            # 实时抓拍列表限制
            real_time_catch_list_count = self.real_time_catch_list.count()
            while real_time_catch_list_count > self.real_time_catch_spin.value():
                self.real_time_catch_list.takeItem(real_time_catch_list_count - 1)
                real_time_catch_list_count -= 1
        except Exception as e:
            print(e)
    def play_video(self):
        if self.playing is not None:
            return
        self.playing = Thread(target=self.playing_video, args=())
        self.playing.start()
    def stop_playing(self):
        if self.playing is not None:
            self.playing = None
    def change_frame(self):
        try:
            if len(self.frame_data_list) == 0:
                return
            current_frame = self.video_process_bar.value()
            max_frame = self.video_process_bar.maximum()
            self.playing_real_time = current_frame == max_frame
            data = self.frame_data_list[current_frame]
            maxData = self.frame_data_list[max_frame]
            frame = data.frame_anno if self.show_box_ckb.isChecked() else data.frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.video_screen.width() - 9, self.video_screen.height() - 9))
            image_height, image_width, image_depth = frame.shape
            frame = QImage(frame.data, image_width, image_height,
                           image_width * image_depth,
                           QImage.Format_RGB888)
            self.video_screen.setPixmap(QPixmap.fromImage(frame))
            current_time_process = second2str(data.time_process)
            max_time_process = second2str(maxData.time_process)
            self.time_process_label.setText(f"{current_time_process}/{max_time_process}")
        except Exception as e:
            print(e)
    def close(self):
        self.close_source()
    def open(self):
        pass
