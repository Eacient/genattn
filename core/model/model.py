import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model

import sys
sys.path.append('.')

from core.model.EPSModel import cam_resnet38d
from core.model.gvae import GMMVAE
from core.functions.spatialBCE import SpatialBCE

def transform_state_dict(path):
    cpkt = torch.load(path, map_location='cpu')
    new_cpkt = {}
    for k,v in cpkt.items():
        if 'fc8' not in k:
            new_cpkt['backbone.'+k]=v
        else:
            new_cpkt[k] = v
    return new_cpkt

def make_prior(cam, cls_label, sp, sal):
    def relu_max_norm(cam, e=1e-5): #to prob but not distribution
        p = F.relu(cam)
        N, C, H, W = p.size()
        p = p / (torch.max(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1) + e)
        return p

    def refine_with_cls_label(prob, cls_label):
        return prob * F.pad(cls_label, [1,0], 'constant',1).unsqueeze(-1).unsqueeze(-1) #[bs, n_class, h, w]
    
    def mean_with_sp(feat, sp):
        #feat [bs,dim,h,w] sp[bs,1,h,w]
        sp_max = sp.max()
        feat_c = feat
        for i in range(int(sp_max)):
            i_mask = (sp == i)
            value_mask = torch.ones_like(i_mask) * i_mask / (i_mask.sum(dim=(2,3), keepdim=True)+1e-8)
            feat_c = feat_c * (~i_mask) + (feat_c * value_mask).sum(dim=(2,3), keepdim=True) * i_mask
        return feat_c

    p = relu_max_norm(cam)
    p = p * cls_label.reshape(*cls_label.shape, 1, 1)
    bg_p = (p > 0) * (1 - p.max(dim=1, keepdim=True)[0])
    p[p < 0.7] = 0
    bg_p[bg_p < 0.6] = 0
    prior = torch.cat([p, bg_p], dim=1)
    # p = mean_with_sp(p, sp)

    # bg_p = 1-p
    
    # p = p * cls_label.reshape(*cls_label.shape, 1, 1)
    # bg_p = bg_p + (1 - p.max(dim=1,keepdim=True)[0] - bg_p)
    # bg_p = bg_p * cls_label.reshape(*cls_label.shape, 1, 1)

    # prior = torch.cat([p, bg_p], dim=1)
    # prior = prior / (prior.sum(dim=1, keepdim=True) + 1e-8)

    return prior #[bs, 2C, h, w]


def make_attn(log_fg_likelihood, cls_label, sp, sal):
    # bs, _, h, w = log_fg_likelihood.shape
    # log_fg_likelihood = log_fg_likelihood.reshape(bs, 2, -1, h, w) #[bs, 2, c, h, w]
    # log_fg_likelihood = 88 - log_fg_likelihood.max(dim=1,keepdim=True)[0] + log_fg_likelihood
    # fg_attn = F.softmax(log_fg_likelihood, dim=1)[:, 0]
    
    # fg_attn = fg_attn * cls_label.reshape(*cls_label.shape, 1, 1)
    bs, _, h, w = log_fg_likelihood.shape
    log_fg_likelihood = 88 - log_fg_likelihood.max(dim=1,keepdim=True)[0] + log_fg_likelihood

    fg_ind = log_fg_likelihood[:, :20]*cls_label.reshape(*cls_label.shape, 1, 1) > log_fg_likelihood[:, 20:]*cls_label.reshape(*cls_label.shape, 1, 1)
    fg_pos = torch.any(fg_ind, dim=1, keepdim=True) #[bs, 1, h, w]
    print(f'binary_fg_ratio={fg_pos.sum()}/{bs*h*w}')


    fg_likelihood = log_fg_likelihood.exp() #[bs, 2C, h, w]
    fg_likelihood = fg_likelihood.reshape(bs, 2, -1, h, w) #[bs, 2, c, h, w]
    fg_likelihood = fg_likelihood * cls_label.reshape(bs, 1, cls_label.size(1), 1, 1)
 
    fg_likelihood_sum = fg_likelihood[:, 0].sum(dim=1, keepdim=True) #[bs, 1, h, w]
    fg_likelihood[:, 1] = fg_likelihood[:, 1] + fg_likelihood_sum - fg_likelihood[:, 0] #[bs, 2, c, h, w]

    fg_attn = fg_likelihood[:, 0] / fg_likelihood.sum(dim=1) #[bs, c, h, w]

    fg_ind = fg_attn > 0.5
    fg_pos = torch.any(fg_ind, dim=1, keepdim=True)
    print(f'fg_ratio={fg_pos.sum()}/{bs*h*w}')


    return fg_attn

def make_attn_post(fg_post, cls_label, sp, sal):
    bs, _, h, w = fg_post.shape
    fg_post = fg_post.reshape(bs, 2, -1, h, w)
    fg_post = (fg_post / (fg_post.sum(dim=1, keepdim=True) + 1e-8))[:, 0]

    fg_post = fg_post * cls_label.reshape(*cls_label.shape, 1, 1)
    # print(fg_post.shape)
    return fg_post


class Net(nn.Module):
    def __init__(self, num_classes=21, use_bg_prob=False, warmup_cam_net=False, pretrained_cam_net='', m=0.999,
                 warmup_gvae=False, feat_dim=1024, embed_dim=128, n_cluster=4, use_learnable_mix_ratio=False,
                 approx_prior=True, wta=False):
        super().__init__()
        self.m = m
        self.warmup_cam_net = warmup_cam_net
        self.warmup_gvae = warmup_gvae
        self.num_classes = num_classes
        self.cam_net = cam_resnet38d(pretrained=True, use_bg_prob=use_bg_prob, num_classes=num_classes)
        if not warmup_cam_net:
            # self.cam_net.load_state_dict(torch.load(pretrained_cam_net,map_location='cpu'))
            self.cam_net.load_state_dict(transform_state_dict('pretrained/voc12_cls.pth'))
            gvae_num_classes = num_classes - 1
            self.gvae = GMMVAE(feat_dim, embed_dim, n_cluster, gvae_num_classes,
                            use_learnable_mix_ratio, approx_prior, wta)

        if not warmup_cam_net and not warmup_gvae:
            self.prior_net = cam_resnet38d(pretrained=True, use_bg_prob=use_bg_prob, num_classes=num_classes)
            # self.prior_net.load_state_dict(torch.load(pretrained_cam_net,map_location='cpu'))
            self.cam_net.load_state_dict(transform_state_dict('pretrained/voc12_cls.pth'))


        self.fix_weights()

    def fix_weights(self):
        self.cam_net.fix_weights()

    def finetune_params(self):
        return ['cam_net.' + s for s  in self.cam_net.finetune_params()]

    def no_weight_decay(self):
        return {}

    @torch.no_grad()
    def _momentume_update_prior_net(self):
        for param_q, param_k in zip(
            self.cam_net.parameters(), self.prior_net.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def discriminate_forward(self, x):
        if self.warmup_gvae:
            feat_dict, cam_scores = self.cam_net.forward_cam(x, dec=True)
            return feat_dict, cam_scores, None, None
        else:
            cam_scores, logit = self.cam_net(x, dec=False)
            with torch.no_grad():
                self._momentume_update_prior_net()
                feat_dict, cam_scores_prior = self.prior_net.forward_cam(x, dec=True)
            return feat_dict, cam_scores_prior, cam_scores, logit
        

    def forward(self, x, cls_label, sp, sal):
        if self.warmup_cam_net:
            cam_scores, logit = self.cam_net(x, dec=False)
            loss_discriminative = F.multilabel_soft_margin_loss(logit, cls_label)
            return cam_scores.detach(), loss_discriminative
        else:
            feat_dict, cam_scores_prior, cam_scores, logit = self.discriminate_forward(x)

            feat_gvae = feat_dict['conv5'].detach()
            prior = make_prior(cam_scores_prior, cls_label, sp, sal)

            loss_recon, kl_inner, kl_prior, log_fg_likelihood, fg_post = self.gvae(feat_gvae, prior)
            # print(f'loss_recon={loss_recon}')
            # print(f'kl_inner={kl_inner}')
            # print(f'kl_prior={kl_prior}')

            fg_attn = make_attn(log_fg_likelihood, cls_label, sp, sal)
            fg_post = make_attn_post(fg_post, cls_label, sp, sal)

            loss_generative = loss_recon + kl_inner + kl_prior
            if self.warmup_gvae:
                return cam_scores_prior.detach(), fg_attn.detach(), fg_post.detach(),\
                      loss_generative
            else:
                loss_discriminative = SpatialBCE(cam_scores, fg_attn, cls_label)
                return cam_scores.detach(), cam_scores_prior.detach(), fg_attn.detach(), fg_post.detach(),\
                      loss_discriminative, loss_generative

    def forward_cam(self, x):
        return self.cam_net.forward_cam(x)

@register_model
def gvae_cam(pretrained=False, **kwargs):
    model =  Net(feat_dim=kwargs['feat_dim'], embed_dim=kwargs['embed_dim'], n_cluster=kwargs['n_cluster'],
                 num_classes=kwargs['num_classes'], 
                 use_learnable_mix_ratio=kwargs['use_learnable_mix_ratio'],
                 wta=kwargs['use_wta'], approx_prior=kwargs['approx_prior'],
                 warmup_cam_net=kwargs['warmup_cam_net'], warmup_gvae=kwargs['warmup_gvae'],
                 pretrained_cam_net=kwargs['cam_net_pretrained']
                 )
    return model

if __name__ == "__main__":
    from core.misc.utils import setup_seed
    setup_seed(0)
    from core.data.VOC12Dataset import VOC12Dataset
    ds = VOC12Dataset('metadata/voc12/train_aug.txt', voc12_root='/home/dogglas/mil/datasets/VOC2012',input_size=224)
    img_id, img, label, sal, seg_label, sp = ds[0]
    img = img.unsqueeze(0).requires_grad_(True)
    label = label.unsqueeze(0).requires_grad_(True)
    sp = sp.unsqueeze(0).unsqueeze(1).requires_grad_(True)
    sal = sal.unsqueeze(0).unsqueeze(1).requires_grad_(True)
    # print(sp.shape)

    img = img.cuda()
    label=label.cuda()
    sal = sal.cuda()
    sp = sp.cuda()

    # model = Net(warmup_cam_net=False, warmup_gvae=True, approx_prior=True, wta=False)
    # model.cam_net.load_state_dict(torch.load('pretrained/cam_net.pth'))
    # model = model.cuda()

    # # cam_scores, loss_discriminative = model(img, label, sp, sal)
    # cam_scores_prior, fg_attn, fg_post, loss_generative = model(img, label, sp, sal)
    # print(loss_generative)

    model = Net(warmup_cam_net=False, warmup_gvae=False, approx_prior=False, wta=True)
    model.cam_net.load_state_dict(torch.load('pretrained/cam_net.pth'))
    model.prior_net.load_state_dict(torch.load('pretrained/cam_net.pth'))
    model = model.cuda()

    cam_scores, cam_scores_prior, fg_attn, fg_post, loss_discriminative, loss_generative = model(img, label, sp, sal)
    print(loss_discriminative, loss_generative)