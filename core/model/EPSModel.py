import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model.backbones.resnet38d import build_resnet38d_backbone, convert_mxnet_to_torch
from timm.models.registry import register_model

class Net(nn.Module):
    def __init__(self, use_bg_prob=False, num_classes=21):
        super(Net, self).__init__()

        self.backbone=build_resnet38d_backbone()
        self.backbone.load_state_dict(convert_mxnet_to_torch())
        
        self.dropout7 = torch.nn.Dropout2d(0.5)

        num_classes = num_classes if use_bg_prob else num_classes-1
        self.fc8 = nn.Conv2d(4096, num_classes, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.fc8.weight)

        self.not_training = [self.backbone.conv1a, self.backbone.b2, self.backbone.b2_1, self.backbone.b2_2]
        self.low_feat = [self.backbone.conv1a, self.backbone.b2, self.backbone.b2_1, self.backbone.b2_2,
                         self.backbone.b3, self.backbone.b3_1, self.backbone.b3_2
                         , self.backbone.b4, self.backbone.b4_1, self.backbone.b4_2, self.backbone.b4_3, self.backbone.b4_4, self.backbone.b4_5
                         , self.backbone.b5, self.backbone.b5_1, self.backbone.b5_2]
        self.fix_weights()

    @torch.jit.ignore
    def finetune_params(self):
        names = []
        for k, _ in self.named_parameters():
            if 'fc8' not in k:
                names.append(k)
        return names

    def fix_weights(self):
        for layer in self.not_training:
            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.requires_grad = False

            elif isinstance(layer, torch.nn.Module):
                for c in layer.children():
                    c.weight.requires_grad = False
                    if c.bias is not None:
                        c.bias.requires_grad = False

        # for layer in self.modules():
        #     if isinstance(layer, torch.nn.BatchNorm2d):
        #         layer.eval()
        #         layer.bias.requires_grad = False
        #         layer.weight.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.fix_weights()
        return
    
    def forward(self, x, dec=False):
        return self.forward_logit(x, dec)

    def forward_logit(self, x, dec=False):
        if not dec:
            x = self.backbone(x)
            x = self.dropout7(x)
            cam = self.fc8(x)
            x = F.adaptive_avg_pool2d(cam, (1,1))
            return cam, x.view(x.size(0), -1)

        else:
            dict = self.backbone(x, dec=True)
            x = self.dropout7(dict['conv6'])
            cam = self.fc8(x)
            x = F.adaptive_avg_pool2d(cam, (1,1))
            return dict, cam, x.view(x.size(0),-1)

    @torch.no_grad()
    def forward_cam(self, x, dec=False):
        if not dec:
            x = self.backbone(x)
            x = F.conv2d(x, self.fc8.weight)
            return x
        else:
            dict = self.backbone(x, dec=True)
            x = F.conv2d(dict['conv6'], self.fc8.weight)
            return dict, x
    
@register_model
def cam_resnet38d(pretrained=True, **kwargs):
    return Net(use_bg_prob=kwargs['use_bg_prob'], num_classes=kwargs['num_classes'])

if __name__ == "__main__":
    cam_bone = cam_resnet38d(use_bg_prob=False, num_classes=21)
    n_parameters = sum(p.numel() for p in cam_bone.parameters())
    print('number of params:', n_parameters)
    # from torchstat import stat
    # print(stat(cam_bone, (3,448,448)))

