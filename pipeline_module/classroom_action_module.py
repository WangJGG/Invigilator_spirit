import torch
from models.classroom_action_classifier import ClassroomActionClassifier
from models.pose_estimator import PnPPoseEstimator
from pipeline_module.core.base_module import TASK_DATA_OK, BaseModule
peep_threshold = -60  # 第一个视频使用阈值-50
class CheatingActionModule(BaseModule):
    class_names = ["正常", "异常", "异常", "异常"]
    use_keypoints = [x for x in range(11)] + [17, 18, 19]
    class_of_passing = [11, 12] # [9, 10, 11, 12]
    class_of_peep = [17, 18]
    class_of_gazing_around = [13, 14, 15, 16]
    def __init__(self, weights, device='cpu', img_size=(480, 640), skippable=True):
        super(CheatingActionModule, self).__init__(skippable=skippable)
        self.weights = weights
        self.classifier = ClassroomActionClassifier(weights, device)
        self.pnp = PnPPoseEstimator(img_size=img_size)
    def process_data(self, data):
        data.num_of_cheating = 0
        data.num_of_normal = 0
        data.num_of_passing = 0
        data.num_of_peep = 0
        data.num_of_gazing_around = 0
        if data.detections.shape[0] > 0:
            data.classes_probs = self.classifier.classify(data.keypoints[:, self.use_keypoints])
            data.raw_best_preds = torch.argmax(data.classes_probs, dim=1)
            data.best_preds = [self.reclassify(idx) for idx in data.raw_best_preds]
            data.classes_names = self.class_names
            data.head_pose = [self.pnp.solve_pose(kp) for kp in data.keypoints[:, 26:94, :2].numpy()]
            data.draw_axis = self.pnp.draw_axis
            data.pred_class_names = [self.class_names[i] for i in data.best_preds]
            data.num_of_normal = data.best_preds.count(0)
            data.num_of_passing = data.best_preds.count(1)
            data.num_of_peep = data.best_preds.count(2)
            data.num_of_gazing_around = data.best_preds.count(3)
            data.num_of_cheating = data.detections.shape[0] - data.num_of_normal
        return TASK_DATA_OK
    def reclassify(self, class_idx):
        if class_idx in self.class_of_passing:
            return 1
        elif class_idx in self.class_of_peep:
            return 2
        elif class_idx in self.class_of_gazing_around:
            return 3
        else:
            return 0
    def open(self):
        super(CheatingActionModule, self).open()
        pass
