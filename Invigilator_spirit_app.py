import sys
from Invigilator_spirit.cheating_detection_app import CheatingDetectionApp
try:
    import Invigilator_spirit_rc
except ImportError:
    pass
import torch
from PyQt5 import QtGui
from PyQt5.QtWidgets import QMainWindow, QApplication
from ui.Invigilator_spirit import Ui_MainWindow as Invigilator_spiritMainWindow# 看ui
from qt_material import apply_stylesheet
torch._C._jit_set_profiling_mode(False)
torch.jit.optimized_execution(False)
class Invigilator_spiritApp(QMainWindow, Invigilator_spiritMainWindow):
    def __init__(self, parent=None):
        super(Invigilator_spiritApp, self).__init__(parent)
        self.setupUi(self)
        self.cheating_detection_widget = CheatingDetectionApp()
        self.cheating_detection_widget.setObjectName("cheating_detection_widget")
        self.tabWidget.addTab(self.cheating_detection_widget, "作弊检测")
    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.cheating_detection_widget.close()
        super(Invigilator_spiritApp, self).closeEvent(a0)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Invigilator_spiritApp()
    apply_stylesheet(app, theme='dark_teal.xml')
    window.show()
    sys.exit(app.exec_())
