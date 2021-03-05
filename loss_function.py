import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models

# Iou loss
class IOU_loss(nn.Module):
     def __init__(self):
         super(IOU_loss, self).__init__()
     def forward(self,result,target):
        soft_output = torch.softmax(result, 1)
        onehot_target = torch.cat(((1.0 - target),target),dim=1)
        tp = torch.sum(soft_output * onehot_target, dim=3)  # result:(8,2,256,256)   target:(8,1,256,256)
        tp1 = torch.sum(tp, dim=2)
        tp2 = torch.sum(tp1, dim=0)

        fn = torch.sum(soft_output * (1.0 - onehot_target), dim=3)
        fn1 = torch.sum(fn, dim=2)
        fn2 = torch.sum(fn1, dim=0)

        fp = torch.sum((1.0 - soft_output) * onehot_target, dim=3)
        fp1 = torch.sum(fp, dim=2)
        fp2 = torch.sum(fp1, dim=0)

        union = tp2 + fn2 + fp2
        iou = tp2 / union
        miou = torch.mean(iou)
        # self.loss_IoU = 1.0 - self.miou
        loss_IoU = - torch.log(miou)
        return loss_IoU

#计算focal loss,这里我们写的只针对而分类任务
class My_focal_loss(torch.nn.Module):
    def __init__(self,gamma=2,alpha=0.25,reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self,input,target):
        pt = torch.sigmoid(input)
        alpha = self.alpha
        gamma = self.gamma
        loss = -alpha*(1-pt)**gamma*target*torch.log(pt)- \
               (1-alpha)*pt**gamma*(1-target)*torch.log(1-pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean()
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


