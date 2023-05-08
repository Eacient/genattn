import torch
import torch.nn.functional as F

def SpatialBCE(cam_scores, fg_attn, cls_label):
    loss_fg = F.multilabel_soft_margin_loss(
        F.adaptive_avg_pool2d(cam_scores * fg_attn, (1,1))
            .view(cam_scores.size(0), -1),
        cls_label
    )
    loss_bg = F.multilabel_soft_margin_loss(
        F.adaptive_avg_pool2d(cam_scores * (1-fg_attn), (1,1))
            .view(cam_scores.size(0), -1),
        torch.zeros_like(cls_label)
    )
    return loss_fg + loss_bg