import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import random

from pathlib import Path

from timm.models import create_model

import sys
sys.path.append('.')

from core.misc.OptimFactory import create_optimizer
import core.misc.utils as utils
from core.misc.utils import NativeScalerWithGradNormCount as NativeScaler

from core.data.VOC12Dataset import bulid_voc_datasets
from core.functions.engine import train_one_epoch
from core.model.model import gvae_cam

def get_args():
    parser = argparse.ArgumentParser('Genseg script', add_help=False)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--accum_iter', default=2, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.set_defaults(debug=True)
    parser.add_argument('--amp', action="store_true")
    # parser.set_defaults(amp=True)
    parser.add_argument('--clip_grad', type=float, default=0.05, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--cam_net_pretrained', default='pretrained/cam_net.pth', type=str)

    parser.add_argument('--use_fixed_var', action='store_true')
    parser.add_argument('--use_learnable_mix_ratio', action='store_true')
    parser.add_argument('--use_wta', action='store_true')
    parser.set_defaults(use_wta=True)
    parser.add_argument('--use_prob', action='store_true')
    parser.set_defaults(use_prob=True)
    parser.add_argument('--return_post', action='store_true')
    parser.add_argument('--use_wcl_loss', action='store_true')
    parser.add_argument('--wcl_k', type=int, default=1)

    parser.add_argument('--recon_weight', type=float, default=1.)
    parser.add_argument('--kl_weight', type=float, default=1.)
    parser.add_argument('--wcl_weight', type=float, default=0.3)

    parser.add_argument('--use_bg_prior', action='store_true')
    parser.add_argument('--bg_thresh', type=float, default=0.5)
    parser.add_argument('--use_sal', action='store_true')
    # parser.set_defaults(use_sal=True)

    parser.add_argument('--train_gen', action='store_true')
    parser.add_argument('--train_cam', action='store_true')
    # parser.set_defaults(train_gen=True)
    parser.set_defaults(train_cam=True)

    # Model parameters
    parser.add_argument('--model', default='gvae_cam', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--feat_dim', type=int, default=1024)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_cluster', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=21)
    parser.add_argument('--temp', type=float, default=1.)


    parser.add_argument('--input_size', default=112, type=int,
                        help='images input size for backbone')
    # Dataset parameters
    parser.add_argument('--name_list', default='metadata/voc12/train_aug.txt', type=str)
    parser.add_argument('--voc12_root', default='/home/dogglas/mil/datasets/VOC2012')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # saving and resume
    parser.add_argument('--wandb_key', default='d32c431927c09a7bf392135ee5b63acaa90f5bee')
    parser.add_argument('--project_name', default='gvae_cam')
    parser.add_argument('--log_dir', default='.')
    parser.add_argument('--output_dir', default='checkpoints_gvae_cam',
                        help='path where to save, empty for no saving')
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    

    # Optimizer parameters
    parser.add_argument('--opt', default='momentum', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=[0.9, 0.999], type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: 0.9, 0.999, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")
    # scheduler
    parser.add_argument('--lr', type=float, default=5e-3, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    # distributed training parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--seed', default=0, type=int)

    return parser.parse_args()


def get_model(args, **kwargs):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        **kwargs,
        feat_dim=args.feat_dim, 
        embed_dim=args.embed_dim, 
        n_cluster=args.n_cluster,
        num_classes=args.num_classes,
        use_learnable_mix_ratio=args.use_learnable_mix_ratio,
        use_learnable_var=not args.use_fixed_var,
        wta=args.use_wta,
        return_post=args.return_post,
        use_one_hot=not args.use_prob,
        wcl_loss=args.use_wcl_loss,
        k=args.wcl_k,
        use_bg_prior=args.use_bg_prior,
        bg_thresh=args.bg_thresh,
        use_sal=args.use_sal,
        temp=args.temp, 
        cam_net_pretrained=args.cam_net_pretrained
    )
    return model

def main(args):
    # if args.debug:
    #     torch.autograd.set_detect_anomaly(True)
    utils.init_distributed_mode(args)
    if not args.amp:
        print('Not using mixed precision training')

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True


    # get dataset
    dataset_train = bulid_voc_datasets(args)

    if args.distributed:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank

        num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
        )
    else:
        num_training_steps_per_epoch = len(dataset_train) // args.batch_size
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    print("Sampler_train = %s" % str(sampler_train))

    if utils.is_main_process():
        log_writer = utils.WandbLogger(args)
    else:
        log_writer = None
    model = get_model(args, pretrained=True)

    eff_batch_size = args.batch_size * args.accum_iter * utils.get_world_size()
    print("accumulate grad iterations: %d" % args.accum_iter)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=eff_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * utils.get_world_size()
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    ### create optimizers for iterate training
    loss_scaler = NativeScaler()

    model_without_ddp.train_cam(create=True)
    optimizer_cam = create_optimizer(args, model_without_ddp)
    model_without_ddp.train_gen(create=True)
    optimizer_gen = create_optimizer(args, model_without_ddp)
    
    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.poly_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if args.resume:
        run_variables={'epoch':0}
        utils.restart_from_checkpoint(args.resume, run_variables=run_variables, 
                                      model=model_without_ddp,
                                      optimizer=optimizer_cam, optimizer2=optimizer_gen,
                                      scaler=loss_scaler)
        args.start_epoch = run_variables['epoch'] + 1
    elif args.auto_resume:
        utils.auto_load_model_iterate(
            args=args, model=model, model_without_ddp=model_without_ddp, 
            optimizer=optimizer_cam, optimizer2=optimizer_gen, loss_scaler=loss_scaler)

    if not args.debug:
        print(f"Start training for {args.epochs} epochs")
    if args.debug:
        args.epochs = args.start_epoch + 2
        print(f"Debug training for 2 epochs")

    start_time = time.time()
    model_without_ddp.grad_all()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if not args.train_cam:
            model.train_gen()
            train_stats = train_one_epoch( #gen epoch
                model, data_loader_train,
                optimizer_gen, device, epoch, loss_scaler,
                args.clip_grad, log_writer=log_writer,
                start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values,
                wd_schedule_values=wd_schedule_values,
                debug=args.debug, amp=args.amp, mode=0, accum_iter=args.accum_iter,
                recon_weight=args.recon_weight, kl_weight=args.kl_weight, wcl_weight=args.wcl_weight
            )
        if not args.train_gen:
            model.train_cam()
            train_stats = train_one_epoch( #cam_epoch
                model, data_loader_train,
                optimizer_cam, device, epoch, loss_scaler,
                args.clip_grad, log_writer=log_writer,
                start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values,
                wd_schedule_values=wd_schedule_values,
                debug=args.debug, amp=args.amp, mode=1, accum_iter=args.accum_iter,
                recon_weight=args.recon_weight, kl_weight=args.kl_weight, wcl_weight=args.wcl_weight
            )
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model_iterate(
                    args=args, model=model, model_without_ddp=model_without_ddp, 
                    optimizer=optimizer_cam, optimizer2=optimizer_gen, loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_gen_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)