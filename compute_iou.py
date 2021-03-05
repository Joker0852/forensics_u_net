import numpy as np
from imageio import imread
from glob import glob
from ntpath import basename
import torch.nn as nn

class my_compute_IoU(nn.Module):
    def __init__(self):
        super(my_compute_IoU, self).__init__()

    def forward(self,path_gt, path_pred):
        iou_list = []
        files = glob(path_gt + '/*.png')  # 按通配符查找文件
        for file in sorted(files):
            filename = basename(file)  # 取文件名
            target = (imread(str(path_gt + filename)) / 255.0).astype(np.float32)
            output = (imread(str(path_pred + filename)) / 255.0)

            # target = 1.0 - target
            # tp_sum = np.sum((1.0 - target) * (1.0 - output))
            tp_sum = np.sum(target * output)
            fn_sum = np.sum(output * (1.0 - target))
            fp_sum = np.sum((1.0 - output) * target)

            iou_idx = tp_sum / (tp_sum + fn_sum + fp_sum)
            iou_list.append(iou_idx)

        iou_mean = np.mean(iou_list)
        return iou_mean
