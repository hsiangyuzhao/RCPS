import torch
import numpy as np
from scipy.ndimage import morphology


def value_check(y_pred, y, nan_val=373.15):
    """
    Hausdorff distance can be nans when the predictions or ground truth are all zeros.
    Before calculating Hausdorff distance, yields predefined HD value according to BraTS rules.
    Args:
        y_pred: np.array, binary, [H, W, ...]
        y: np.array, binary, [H, W, ...]
        nan_val: predefined HD val, in BraTS rules set as the largest HD value in the competition
    """
    if y_pred.max() == y.max() == 0:
        return 0.0, False
    elif y.max() == 0 and y_pred.max() > 0:
        return nan_val, False
    elif y_pred.max() == 0 and y.max() > 0:
        return nan_val, False
    else:
        return 0.0, True


def surface_distance(y_pred, y, sampling=1, connectivity=1):
    """
    Compute the surface distance between predictions and ground truth
    https://mlnotebook.github.io/post/surface-distance-function/
    Args:
        y_pred: np.array, binary, [H, W, ...]
        y: np.array, binary, [H, W, ...]
        sampling: voxel spacing
        connectivity: see scipy.morphology.generate_binary_structure
    """
    y_pred = np.atleast_1d(y_pred.astype(bool))
    y = np.atleast_1d(y.astype(np.bool))

    kernel = morphology.generate_binary_structure(y_pred.ndim, connectivity)

    y_pred_border = y_pred ^ morphology.binary_erosion(y_pred, kernel)
    y_border = y ^ morphology.binary_erosion(y, kernel)

    # distance transform of pred and gt
    # TODO: very slow when calculating on large arrays, any idea to accelerate it?
    dta = morphology.distance_transform_edt(~y_pred_border, sampling)
    dtb = morphology.distance_transform_edt(~y_border, sampling)

    sds = np.concatenate([np.ravel(dta[y_border != 0]), np.ravel(dtb[y_pred_border != 0])])

    return sds


def hausdorff_distance(y_pred, y, directed=True, percentile=None, sampling=1):
    """
    Compute the Hausdorff distance between predictions and ground truth
    Args:
        y_pred: np.array, binary, [H, W, ...]
        y: np.array, binary, [H, W, ...]
        directed: bool, whether to calculate directed HD
        precentile: int, whether to calculate percentile HD
        sampling: voxel spacing
    """
    sd_p2g = surface_distance(y_pred, y, sampling=sampling)
    if not directed:
        if percentile:
            hd = np.percentile(sd_p2g, percentile)
        else:
            hd = np.max(sd_p2g)
    else:
        sd_g2p = surface_distance(y, y_pred, sampling=sampling)
        sd_stack = np.hstack([sd_p2g, sd_g2p])
        if percentile:
            hd = np.percentile(sd_stack, percentile)
        else:
            hd = np.max(sd_stack)
    return hd


def calculate_Dice_score(y_pred, y):
    """
    Compute the Dice score between predictions and ground truth
    Args:
        y_pred: torch.tensor, binary, [B, 1, H, W, ...]
        y: torch.tensor, binary, [B, 1, H, W, ...]
    """
    assert y_pred.shape[1] == 1 and y.shape[1] == 1, 'The calculator works for binary case, ' \
                                                     'please check the input shape matches [B, 1, H, ...]'
    n_len = len(y_pred.shape)
    reduce_axis = list(range(1, n_len))

    intersection = torch.sum(y * y_pred, dim=reduce_axis)  # intersection of dice, shape [B, ]

    y_o = torch.sum(y, dim=reduce_axis)
    y_pred_o = torch.sum(y_pred, dim=reduce_axis)
    denominator = y_o + y_pred_o  # shape [B, ]

    dice_raw = (2.0 * intersection) / denominator  # dice score, may contain torch.nans as denominator can be zero

    # if denominator is zero, meaning that the predictions and ground-truth are all zeros, Dice should be 1.0
    dice = torch.where(denominator == 0, torch.tensor(1.0, device=y_o.device), dice_raw)
    return dice


def calculate_avg_surface_distance(y_pred, y):
    """
    Compute the average surface distance between predictions and ground truth
    Args:
        y_pred: torch.tensor, binary, [B, 1, H, W, ...]
        y: torch.tensor, binary, [B, 1, H, W, ...]
    """
    assert y_pred.shape[1] == 1 and y.shape[1] == 1, 'The calculator works for binary case, ' \
                                                     'please check the input shape matches [B, 1, H, ...]'
    batch_size = y_pred.shape[0]
    y_pred_numpy = y_pred.squeeze(1).cpu().numpy()  # [B, H, W, D]
    y_numpy = y.squeeze(1).cpu().numpy()  # [B, H, W, D]
    asd_array = np.empty((batch_size,))
    for idx in range(y_pred.shape[0]):
        pred = y_pred_numpy[idx]
        gt = y_numpy[idx]
        asd_array[idx,] = surface_distance(pred, gt).mean()
    return torch.from_numpy(asd_array)


def calculate_Hasudorff_distance(y_pred, y, directed=True, percentile=None):
    """
    Compute the Hausdorff distance between predictions and ground truth
    Args:
        y_pred: np.array, binary, [H, W, ...]
        y: np.array, binary, [H, W, ...]
        directed: bool, whether to calculate directed HD
        precentile: int, whether to calculate percentile HD
    """
    assert y_pred.shape[1] == 1 and y.shape[1] == 1, 'The calculator works for binary case, ' \
                                                     'please check the input shape matches [B, 1, H, ...]'
    batch_size = y_pred.shape[0]
    y_pred_numpy = y_pred.squeeze(1).cpu().numpy()  # [B, H, W, D]
    y_numpy = y.squeeze(1).cpu().numpy()  # [B, H, W, D]
    hd_array = np.empty((batch_size,))
    for idx in range(y_pred.shape[0]):
        pred = y_pred_numpy[idx]
        gt = y_numpy[idx]
        value, not_nan = value_check(pred, gt)
        if not_nan:
            hd_array[idx, ] = hausdorff_distance(pred, gt, directed=directed, percentile=percentile)
        else:
            hd_array[idx, ] = value
    return torch.from_numpy(hd_array)
