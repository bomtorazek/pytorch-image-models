#!/usr/bin/env python
"""PyTorch Inference Script

An example inference script that outputs top-k class ids for images in a folder into a csv.

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import os
import time
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from shutil import copyfile
from timm.models import create_model, apply_test_time_pool
from timm.data import Dataset, create_loader, resolve_data_config
from timm.utils import AverageMeter, setup_default_logging

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('inference')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--output_dir', metavar='DIR', default='./',
                    help='path to output files')
parser.add_argument('--model', '-m', metavar='MODEL', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-test-pool', dest='no_test_pool', action='store_true',
                    help='disable test time pool')
parser.add_argument('--topk', default=1, type=int,
                    metavar='N', help='Top-k to output to CSV')
parser.add_argument('--thresh', default=0.5, type=float,
                    metavar='N', help='')


def main():
    setup_default_logging()
    args = parser.parse_args()
    # might as well try to do something useful...
    args.pretrained = args.pretrained or not args.checkpoint

    # create model
    model = create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=3,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint)

    _logger.info('Model %s created, param count: %d' %
                 (args.model, sum([m.numel() for m in model.parameters()])))

    config = resolve_data_config(vars(args), model=model)
    model, test_time_pool = (model, False) if args.no_test_pool else apply_test_time_pool(model, config)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    else:
        model = model.cuda()

    loader = create_loader(
        Dataset(args.data, train_mode= 'test', fold_num= -1),
        input_size=config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=True,
        interpolation=config['interpolation'],
        mean=config['mean'],
        std=config['std'],
        num_workers=args.workers,
        crop_pct=1.0 if test_time_pool else config['crop_pct'])

    model.eval()

#     k = min(args.topk, args.num_classes)
    batch_time = AverageMeter()
    end = time.time()
    topk_ids = []
    name_list = []
    sig_list = []
    logits_list = []
    m = torch.nn.Sigmoid()
    with torch.no_grad():
        for batch_idx, (input, _,) in enumerate(loader):
            input = input.cuda()
            labels = model(input)
            logits_list.append(labels)
            sigmoided = m(labels)
            sig_list.append(np.expand_dims(sigmoided[:,1].cpu().numpy(), axis = 1))
#             topk = labels.topk(k)[1]
#             topk_ids.append(topk.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                _logger.info('Predict: [{0}/{1}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    batch_idx, len(loader), batch_time=batch_time))

#     topk_ids = np.concatenate(topk_ids, axis=0).squeeze()
#     logits = torch.cat(logits_list).cuda()
#     temperature = nn.Parameter(torch.ones(1) * args.te).to(torch.device('cuda') ).detach().requires_grad_(False)
#     logits = logits/temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
#     temp_sigmoided =  m(logits)[:,1].detach().cpu().numpy()
    
    sig_list = np.vstack(sig_list)
    name_list = loader.dataset.filenames(basename=True)
    
    real_sigmoid = sig_list.squeeze()
#     real_sigmoid = temp_sigmoided
    real_pred = ((sig_list >= args.thresh)*1).squeeze()

    name_pred_dict = {}
    for idx in range(len(name_list)):
        name_pred_dict[name_list[idx]] = (real_pred[idx], real_sigmoid[idx])
    
    args.output_dir = args.checkpoint.replace(args.checkpoint.split('/')[-1], "")
    with open(os.path.join(args.output_dir, './prediction.tsv'), 'w') as out_file:
#         filenames_int = [int(f.split('.')[0]) for f in filenames]
#         for name, topk in zip(filenames_int, topk_ids):
#             print(name,topk)
#             i = i+1
#             if i == 10:
#                 break
#         idx = np.argsort(filenames_int)
#         topk_ids = topk_ids[idx]
        for name in name_list:
            out_file.write('{}\n'.format(str(name_pred_dict[name][0])))
    with open(os.path.join(args.output_dir, './probability.tsv'), 'w') as out_file:
        for name in name_list:
            out_file.write('{}\n'.format(name_pred_dict[name][1]))
            
    copyfile(os.path.join(args.output_dir, './prediction.tsv'), '/home/workspace/user-workspace/prediction/'+'prediction_153_'+ args.checkpoint.split('/')[-2] +'.tsv')
    
if __name__ == '__main__':
    main()
