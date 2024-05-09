# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com), Haoyi Zhu
# -----------------------------------------------------
import cv2
import numpy as np
try:
    from alphapose.bbox import (_box_to_center_scale, _center_scale_to_box)
    from alphapose.transforms import (get_affine_transform, im_to_torch)
except ImportError:
    from utils.alphapose.bbox import (_box_to_center_scale, _center_scale_to_box)
    from utils.alphapose.transforms import (get_affine_transform, im_to_torch)
class SimpleTransform(object):
    def __init__(self, scale_factor, add_dpg,
                 input_size, output_size, rot, sigma,
                 gpu_device=None, loss_type='MSELoss'):
        self._scale_factor = scale_factor
        self._rot = rot
        self._add_dpg = add_dpg
        self._gpu_device = gpu_device
        self._input_size = input_size
        self._heatmap_size = output_size
        self._sigma = sigma
        self._loss_type = loss_type
        self._aspect_ratio = float(input_size[1]) / input_size[0]  # w / h
        self._feat_stride = np.array(input_size) / np.array(output_size)
        self.pixel_std = 1
    def test_transform(self, src, bbox):
        xmin, ymin, xmax, ymax = bbox
        center, scale = _box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio)
        scale = scale * 1.0
        input_size = self._input_size
        inp_h, inp_w = input_size
        trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
        img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        bbox = _center_scale_to_box(center, scale)
        img = im_to_torch(img)
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)
        return img, bbox
