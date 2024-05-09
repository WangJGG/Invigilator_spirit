import os
from time import strftime, localtime
import cv2
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QListWidgetItem, QWidget, QListWidget
from ui.cheating_list_item import Ui_CheatingListItem
from ui.real_time_catch import Ui_RealTimeCatch
from utils.common import second2str
from utils.img_cropper import CropImage
cnt=0
class FrameData(QListWidgetItem):
    def __init__(self, list_widget: QListWidget, data, filter_idx=0):
        super(FrameData, self).__init__()
        self.list_widget = list_widget
        self.data_ = data
        self.frame_num = data.frame_num
        self.widget = FrameData.Widget(list_widget)
        self.time_process = second2str(data.time_process)
        color1 = '#ff0000' if data.num_of_passing > 0 else '#ffffff'
        color2 = '#ff0000' if data.num_of_peep > 0 else '#ffffff'
        color3 = '#ff0000' if data.num_of_gazing_around > 0 else '#ffffff'
        self.str1 = f"时间[{self.time_process}]"
        self.widget.lbl1.setText(self.str1)
        self.str2 = "可能作弊行为："\
                    f"<span style=\" color: {color1};\">传纸条: {data.num_of_passing}</span> " \
                    f"<span style=\" color: {color2};\">低头偷看: {data.num_of_peep}</span> " \
                    f"<span style=\" color: {color3};\">东张西望: {data.num_of_gazing_around}</span>"
        self.widget.lbl2.setText(self.str2)
        self.setSizeHint(QSize(500, 81))
    def add_item(self):
        size = self.sizeHint()
        self.list_widget.addItem(self)
        self.widget.setSizeIncrement(size.width(), size.height())
        self.list_widget.setItemWidget(self, self.widget)
    class Widget(QWidget, Ui_CheatingListItem):
        def __init__(self, parent=None):
            super(FrameData.Widget, self).__init__(parent)
            self.setupUi(self)
class RealTimeCatchItem(QListWidgetItem):
    def __init__(self, list_widget: QListWidget, img, detection, time_process, cheating_type, frame_num):
        super(RealTimeCatchItem, self).__init__()
        self.list_widget = list_widget
        self.widget = RealTimeCatchItem.Widget(list_widget)
        self.setSizeHint(QSize(200, 200))
        self.img = img
        self.time_process = time_process
        self.cheating_type = cheating_type
        self.frame_num = frame_num
        self.detection = detection
    def add_item(self):
        size = self.sizeHint()
        self.list_widget.insertItem(0, self)
        self.widget.setSizeIncrement(size.width(), size.height())
        self.list_widget.setItemWidget(self, self.widget)
        catch_img = self.widget.catch_img
        frame = CropImage.crop(self.img, self.detection, 1, catch_img.width(), catch_img.height())
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_height, image_width, image_depth = frame.shape
        frame = QImage(frame.data, image_width, image_height,
                       image_width * image_depth,
                       QImage.Format_RGB888)
        name = os.getcwd()
        print('name1=', name)
        timestamp = strftime('%Y%m%d%H%M%S', localtime())
        name = f"{name}\\resource\pic\\{timestamp}.jpg"
        print('name2=', name)
        flag = frame.save(name, 'JPG', 100)
        print('flag=', flag)
        self.widget.catch_img.setPixmap(QPixmap.fromImage(frame))
        self.widget.time_lbl.setText(f'{second2str(self.time_process)}')
    class Widget(QWidget, Ui_RealTimeCatch):
        def __init__(self, parent=None):
            super(RealTimeCatchItem.Widget, self).__init__(parent)
            self.setupUi(self)
class VideoSourceItem(QListWidgetItem):
    def __init__(self, list_widget, name, src, ico_src=":/videos/web-camera.ico"):
        super(VideoSourceItem, self).__init__()
        icon = QIcon()
        icon.addPixmap(QPixmap(ico_src), QIcon.Normal, QIcon.Off)
        self.setIcon(icon)
        self.setText(name)
        self.src = src
        self.list_widget = list_widget
    def add_item(self):
        self.list_widget.addItem(self)