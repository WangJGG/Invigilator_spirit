# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------
#
import cv2
import numpy as np
import torch
from torch.nn import functional as F
def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img
def to_torch(ndarray):
    # numpy.ndarray => torch.Tensor
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray
def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    """Rotate the point by `rot_rad` degree."""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def flip(x):
    assert (x.dim() == 3 or x.dim() == 4)
    dim = x.dim() - 1

    return x.flip(dims=(dim,))
def heatmap_to_coord_simple_regress(preds, bbox, hm_shape, norm_type, hms_flip=None):
    def integral_op(hm_1d):
        if hm_1d.device.index is not None:
            hm_1d = hm_1d * torch.cuda.comm.broadcast(torch.arange(hm_1d.shape[-1]).type(
                torch.cuda.FloatTensor), devices=[hm_1d.device.index])[0]
        else:
            hm_1d = hm_1d * torch.arange(hm_1d.shape[-1]).type(torch.FloatTensor)
        return hm_1d

    if preds.dim() == 3:
        preds = preds.unsqueeze(0)
    hm_height, hm_width = hm_shape
    num_joints = preds.shape[1]

    pred_jts, pred_scores = _integral_tensor(preds, num_joints, False, hm_width, hm_height, 1, integral_op, norm_type)
    pred_jts = pred_jts.reshape(pred_jts.shape[0], num_joints, 2)

    if hms_flip is not None:
        if hms_flip.dim() == 3:
            hms_flip = hms_flip.unsqueeze(0)
        pred_jts_flip, pred_scores_flip = _integral_tensor(hms_flip, num_joints, False, hm_width, hm_height, 1, integral_op, norm_type)
        pred_jts_flip = pred_jts_flip.reshape(pred_jts_flip.shape[0], num_joints, 2)

        pred_jts = (pred_jts + pred_jts_flip) / 2
        pred_scores = (pred_scores + pred_scores_flip) / 2

    ndims = pred_jts.dim()
    assert ndims in [2, 3], "Dimensions of input heatmap should be 3 or 4"
    if ndims == 2:
        pred_jts = pred_jts.unsqueeze(0)
        pred_scores = pred_scores.unsqueeze(0)

    coords = pred_jts.cpu().numpy()
    coords = coords.astype(np.float32)
    pred_scores = pred_scores.cpu().numpy()
    pred_scores = pred_scores.astype(np.float32)

    coords[:, :, 0] = (coords[:, :, 0] + 0.5) * hm_width
    coords[:, :, 1] = (coords[:, :, 1] + 0.5) * hm_height

    preds = np.zeros_like(coords)
    # transform bbox to scale
    xmin, ymin, xmax, ymax = bbox
    w = xmax - xmin
    h = ymax - ymin
    center = np.array([xmin + w * 0.5, ymin + h * 0.5])
    scale = np.array([w, h])
    # Transform back
    for i in range(coords.shape[0]):
        for j in range(coords.shape[1]):
            preds[i, j, 0:2] = transform_preds(coords[i, j, 0:2], center, scale,
                                               [hm_width, hm_height])

    if preds.shape[0] == 1:
        preds = preds[0]
        pred_scores = pred_scores[0]
    return preds, pred_scores
def _integral_tensor(preds, num_joints, output_3d, hm_width, hm_height, hm_depth, integral_operation, norm_type='softmax'):
    # normalization
    preds = preds.reshape((preds.shape[0], num_joints, -1))
    preds = norm_heatmap(norm_type, preds)

    # get heatmap confidence
    if norm_type == 'sigmoid':
        maxvals, _ = torch.max(preds, dim=2, keepdim=True)
    else:
        maxvals = torch.ones(
            (*preds.shape[:2], 1), dtype=torch.float, device=preds.device)

    # normalized to probability
    heatmaps = preds / preds.sum(dim=2, keepdim=True)
    heatmaps = heatmaps.reshape(
        (heatmaps.shape[0], num_joints, hm_depth, hm_height, hm_width))

    # The edge probability
    hm_x = heatmaps.sum((2, 3))
    hm_y = heatmaps.sum((2, 4))
    hm_z = heatmaps.sum((3, 4))

    hm_x = integral_operation(hm_x)
    hm_y = integral_operation(hm_y)
    hm_z = integral_operation(hm_z)

    coord_x = hm_x.sum(dim=2, keepdim=True)
    coord_y = hm_y.sum(dim=2, keepdim=True)
    coord_z = hm_z.sum(dim=2, keepdim=True)

    coord_x = coord_x / float(hm_width) - 0.5
    coord_y = coord_y / float(hm_height) - 0.5
    if output_3d:
        coord_z = coord_z / float(hm_depth) - 0.5
        pred_jts = torch.cat((coord_x, coord_y, coord_z), dim=2)
        pred_jts = pred_jts.reshape((pred_jts.shape[0], num_joints * 3))
    else:
        pred_jts = torch.cat((coord_x, coord_y), dim=2)
        pred_jts = pred_jts.reshape((pred_jts.shape[0], num_joints * 2))
    return pred_jts, maxvals.float()
def norm_heatmap(norm_type, heatmap):
    # Input tensor shape: [N,C,...]
    shape = heatmap.shape
    if norm_type == 'softmax':
        heatmap = heatmap.reshape(*shape[:2], -1)
        # global soft max
        heatmap = F.softmax(heatmap, 2)
        return heatmap.reshape(*shape)
    elif norm_type == 'sigmoid':
        return heatmap.sigmoid()
    elif norm_type == 'divide_sum':
        heatmap = heatmap.reshape(*shape[:2], -1)
        heatmap = heatmap / heatmap.sum(dim=2, keepdim=True)
        return heatmap.reshape(*shape)
    else:
        raise NotImplementedError
def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    target_coords[0:2] = affine_transform(coords[0:2], trans)
    return target_coords
def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans
def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

