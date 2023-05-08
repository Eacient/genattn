import math
import sys
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from misc.utils import SmoothedValue, MetricLogger, miou
import misc.utils as utils



def label_from_prob(prob):
    p = prob
    bg = torch.ones_like(p[:, :1, :, :]) * 0.2
    p = torch.cat([bg,p], dim=1)
    return p.max(dim=1)[1]

def label_from_post(post):
    p = post
    return p.max(dim=1)[1]


def label_from_scores(scores):
    # p = F.softmax(scores, dim=1)
    p = scores
    return p.max(dim=1)[1]

class empty_env():
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None,
                    debug=False, amp=False, mode=0, accum_iter=1,
                    recon_weight=1, kl_weight=1):
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch_gen: [{}]'.format(epoch) if mode == 0 else 'Epoch_cam: [{}]'.format(epoch)
    print_freq = 10 if not debug else 1

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if debug and step > 2:
            break
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if step % accum_iter == 0:
            lr_update(lr_schedule_values, wd_schedule_values, optimizer, it)
        update_grad = (step + 1) % accum_iter == 0 or step + 1 == len(data_loader)


        img_id, img, label, sal, seg, sp = batch
        img = img.to(device, non_blocking=True) #[bs, 3, h, w]
        label = label.to(device, non_blocking=True) #[bs, n_class]
        sal = sal.to(device, non_blocking=True).unsqueeze(1) #[bs, 1, h, w]
        sp = sp.to(device, non_blocking=True).unsqueeze(1) #[bs,1,h, w]

        grad_env = torch.cuda.amp.autocast if amp else empty_env

        with grad_env():
            
            loss = None
        # torch.autograd.backward(loss, retain_graph=True)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        grad_norm, loss_scale_value = loss_backward(model, optimizer, loss, max_norm, amp, loss_scaler, accum_iter, update_grad)

        if step % print_freq == 0:
            post_label = label_from_post(F.interpolate(cls_post, img.shape[-2:])).cpu()
            cam_label = label_from_post(F.interpolate(cam_prob, img.shape[-2:])).cpu()
            seg_acc, seg_recall, seg_precision, miu_post, fwavacc = miou(post_label, seg, num_classes=21)
            seg_acc, seg_recall, seg_precision, miu_cam, fwavacc = miou(cam_label, seg, num_classes=21)
            metric_logger.update(miu_post=miu_post)
            metric_logger.update(miu_cam=miu_cam)
            if log_writer is not None:
                log_writer.update(miu_post=miu_post, miu_cam=miu_cam,
                                   orig=(img[0].detach().cpu(), seg[0].numpy(), post_label[0].numpy(), cam_label[0].numpy()))

        #---------logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        metric_logger.update(recon_loss=recon_loss_value)
        metric_logger.update(kl_loss=kl_loss_value)
        metric_logger.update(wcl_loss=wcl_loss_value)
        metric_logger.update(cls_loss=cls_loss_value)
        metric_logger.update(aux_loss=aux_loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        metric_logger.update(grad_norm=grad_norm)
        max_lr = 0.
        for group in optimizer.param_groups:
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

        if log_writer is not None:
            log_writer.update(loss=loss_value, 
                              recon_loss=recon_loss_value,
                              kl_loss=kl_loss_value,
                              wcl_loss=wcl_loss_value,
                              cls_loss=cls_loss_value,
                              aux_loss=aux_loss_value,
                              loss_scale=loss_scale_value,
                              grad_norm=grad_norm,
                              lr=max_lr)

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger, "\n")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def lr_update(lr_schedule_values, wd_schedule_values, optimizer, it):
    if lr_schedule_values is not None or wd_schedule_values is not None:
        for i, param_group in enumerate(optimizer.param_groups):
            if lr_schedule_values is not None:
                param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
            if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_schedule_values[it]

def loss_backward(model, opt, loss, max_norm=None, amp=False, loss_scaler=None, accum_iter=1, update_grad=True):
    loss /= accum_iter
    if amp:
        assert loss_scaler is not None
        is_second_order = hasattr(opt, 'is_second_order') and opt.is_second_order
        grad_norm = loss_scaler(loss, opt, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=update_grad)
        loss_scale_value = loss_scaler.state_dict()['scale']
    else:
        if update_grad:
            loss.backward()
            if max_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            else:
                grad_norm = utils.get_grad_norm_(model.parameters())
            opt.step()
        else:
            grad_norm = None
        loss_scale_value = 1
    if update_grad:
        opt.zero_grad()
    if grad_norm is None:
        grad_norm = 0
    return grad_norm, loss_scale_value

