import math
import numpy as  np
import torch
default_focus = torch.tensor([1176.03538718, 165.21773522])
class CheatingActionAnalysis:
    @staticmethod
    def stretch_out_degree(keypoints, left=True, right=True, focus=None):
        if focus is None:
            shoulder_vec = keypoints[:, 6] - keypoints[:, 5]
        else:
            shoulder_vec = (keypoints[:, 6] + keypoints[:, 5]) / 2 - focus
        result = []
        if left:
            arm_vec = keypoints[:, 5] - keypoints[:, 7]
            forearm_vec = keypoints[:, 7] - keypoints[:, 9]
            _results = torch.hstack([torch.cosine_similarity(shoulder_vec, arm_vec).unsqueeze(1),
                                     torch.cosine_similarity(shoulder_vec, forearm_vec).unsqueeze(1),
                                     torch.cosine_similarity(arm_vec, forearm_vec).unsqueeze(1)])
            result.append(_results)
        if right:
            shoulder_vec = -shoulder_vec
            arm_vec = keypoints[:, 6] - keypoints[:, 8]
            forearm_vec = keypoints[:, 8] - keypoints[:, 10]
            _results = torch.hstack([torch.cosine_similarity(shoulder_vec, arm_vec).unsqueeze(1),
                                     torch.cosine_similarity(shoulder_vec, forearm_vec).unsqueeze(1),
                                     torch.cosine_similarity(arm_vec, forearm_vec).unsqueeze(1)])
            result.append(_results)
        return result
    @staticmethod
    def is_stretch_out(degree, threshold=None, dim=1):
        if threshold is None:
            threshold = torch.tensor([0.8, 0.8, 0.5])
        return torch.all(degree > threshold, dim=dim)
    @staticmethod
    def is_passing(keypoints):
        irh = CheatingActionAnalysis.is_raise_hand(keypoints)
        stretch_out_degree_L, stretch_out_degree_R = CheatingActionAnalysis.stretch_out_degree(keypoints)
        isoL = CheatingActionAnalysis.is_stretch_out(stretch_out_degree_L)
        isoR = CheatingActionAnalysis.is_stretch_out(stretch_out_degree_R)
        left_pass = isoL & ~irh[:, 0]
        left_pass_value = torch.zeros_like(left_pass, dtype=int)
        left_pass_value[left_pass] = 1
        right_pass = isoR & ~irh[:, 1]
        right_pass_value = torch.zeros_like(right_pass, dtype=int)
        right_pass_value[right_pass] = -1
        return left_pass_value + right_pass_value
    @staticmethod
    def is_raise_hand(keypoints):
        return keypoints[:, [17], 1] > keypoints[:, [9, 10], 1]
depth_correction_factor = 0.28550474951641663
def turn_head_angle(rvec, tvec):
    x = tvec[0][0]
    depth = tvec[2][0] * depth_correction_factor
    if depth < 0:
        depth, x = -depth, -x
    right_front_angle = math.acos(depth / math.sqrt((x * x + depth * depth)))
    if x < 0:
        return rvec[1][0] + right_front_angle
    else:
        return rvec[1][0] - right_front_angle
def turn_head_angle_yaw(yaw, tvec):
    x = tvec[0][0]
    depth = tvec[2][0] * depth_correction_factor
    if depth < 0:
        depth, x = -depth, -x
    right_front_angle = math.acos(depth / math.sqrt((x * x + depth * depth))) * 57.3
    print(x, depth, right_front_angle, yaw)
    if x < 0:
        return yaw - right_front_angle
    else:
        return yaw + right_front_angle
if __name__ == '__main__':
    p1, p2, p3, p4 = (74, 182), (271, 179), (386, 359), (757, 268)
    A = np.array([[p2[1] - p1[1], p1[0] - p2[0]],
                  [p4[1] - p3[1], p3[0] - p4[0]]])
    b = np.array([(p1[0] - p2[0]) * p1[1] - (p1[1] - p2[1]) * p1[0],
                  (p3[0] - p4[0]) * p3[1] - (p3[1] - p4[1]) * p3[0]])
    print(np.linalg.solve(A, b))
