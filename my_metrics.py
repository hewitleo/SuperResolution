import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity


def calculate_psnr(original, predicted): #my code
    assert original.shape == predicted.shape
    max_value = np.max(original)
    dim_xyz = original.shape[0] * original.shape[1] * original.shape[2]
    mse = (1 / dim_xyz) * (np.sum((original.reshape(-1) - predicted.reshape(-1))**2))

    if mse ==0:
        return float('inf')
    
    psnr = 10 * np.log10( max_value**2 / mse)
    return psnr


def compare_sam(x_true, x_pred): #copied
    """
    :param x_true: 高光谱图像：格式：(H, W, C)
    :param x_pred: 高光谱图像：格式：(H, W, C)
    :return: 计算原始高光谱数据与重构高光谱数据的光谱角相似度
    """
    num = 0
    sum_sam = 0
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    for x in range(x_true.shape[0]):
        for y in range(x_true.shape[1]):
            tmp_pred = x_pred[x, y].ravel()
            tmp_true = x_true[x, y].ravel()
            if np.linalg.norm(tmp_true) != 0 and np.linalg.norm(tmp_pred) != 0:
                sum_sam += np.arccos(
                    np.inner(tmp_pred, tmp_true) / (np.linalg.norm(tmp_true) * np.linalg.norm(tmp_pred)))
                num += 1
    sam_deg = (sum_sam / num) * 180 / np.pi
    return sam_deg



# def compare_mssim(x_true, x_pred, data_range, multidimension): #copied
#     """

#     :param x_true:
#     :param x_pred:
#     :param data_range:
#     :param multidimension:
#     :return:
#     """
#     mssim = [structural_similarity(X=x_true[:, :, i], Y=x_pred[:, :, i], data_range=data_range, multidimension=multidimension)
#             for i in range(x_true.shape[2])]

#     return np.mean(mssim)