from net.sanet import SANet
import time
from dataset.collate import train_collate, test_collate, eval_collate
from dataset.bbox_reader import BboxReader
from utils.util import Logger
from config import train_config, data_config, net_config, config
from net.model_config import VARIANTS, build_model_config, checkpoint_model_config
import pprint
from torch.utils.data import DataLoader, StackDataset, ConcatDataset
from torch.autograd import Variable
import torch
import numpy as np
import argparse
import os
import sys
from tqdm import tqdm
import random
import traceback
import json
from pathlib import Path
from net.training_setup import (TRAINING_FIELDS, resolve_training_settings,
    resolve_model_for_training, load_training_weights, install_health_hooks,
    configure_training_experiment)
from torch.utils.tensorboard import SummaryWriter
import shutil
import pandas as pd

torch.backends.cudnn.benchmark = True
scaler = torch.cuda.amp.GradScaler()

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
this_module = sys.modules[__name__]

parser = argparse.ArgumentParser(description='PyTorch Detector')
parser.add_argument('--net', '-m', metavar='NET', default=train_config['net'],
                    help='neural net')
parser.add_argument('--variant', choices=VARIANTS, default=None,
                    help='architecture ablation; default baseline, or restore from --ckpt')
parser.add_argument('--epochs', default=None, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=train_config['batch_size'], type=int, metavar='N',
                    help='batch size')
parser.add_argument('--epoch-rcnn', default=None, type=int, metavar='NR',
                    help='number of epochs before training rcnn')
checkpoint_group = parser.add_mutually_exclusive_group()
checkpoint_group.add_argument('--ckpt', default=train_config['initial_checkpoint'], type=str, metavar='CKPT',
                    help='resume model, optimizer and epoch')
checkpoint_group.add_argument('--init-from', type=str,
                    help='load weights only; start epoch 1 with a new optimizer')
parser.add_argument('--reset-heads', action='store_true', help='with --init-from, reinitialize RPN and RCNN heads')
parser.add_argument('--head-activation', choices=('relu', 'leaky_relu'), default=None)
parser.add_argument('--auxiliary-losses', choices=('on', 'off'), default=None,
                    help='off bypasses KL/low-rank branches; saved in model_config')
parser.add_argument('--scale-augmentation', choices=('on', 'off'), default=None,
                    help='control crop scale augmentation; saved in model_config')
parser.add_argument('--max-grad-norm', type=float, default=None, help='gradient clipping; zero disables it')
parser.add_argument('--stop-after-epoch', type=int, help='stop early without compressing the --epochs LR schedule')
parser.add_argument('--report-head-health', action='store_true', help='log activations/gradients every 200 training batches')
parser.add_argument('--rpn-train-pre-topk', type=int, default=None, help='train/valid candidates before NMS; zero means unlimited')
parser.add_argument('--rpn-train-post-topk', type=int, default=None, help='train/valid candidates after NMS; zero means unlimited')
parser.add_argument('--optimizer', default=None, type=str, metavar='SPLIT',
                    help='which split set to use')
parser.add_argument('--init-lr', default=None, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=None, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', default=None, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--epoch-save', default=train_config['epoch_save'], type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--out-dir', default=train_config['out_dir'], type=str, metavar='OUT',
                    help='directory to save results of this training')
parser.add_argument('--train-set', default=train_config['train_set_list'], nargs='+', type=str,
                    help='train set paths list')
parser.add_argument('--val-set', default=train_config['val_set_list'], nargs='+', type=str,
                    help='val set paths list')
parser.add_argument('--train-anno', default=config['train_anno'], nargs='+',
                    help='training annotation CSVs, one per source dataset')
parser.add_argument('--val-anno', default=config['val_anno'], nargs='+',
                    help='validation annotation CSVs, one per dataset')
# parser.add_argument('--train-data-dir', default=train_config['train_dir'], type=str, metavar='OUT',
#                     help='path to load data')
# parser.add_argument('--val-data-dir', default=train_config['val_dir'], type=str, metavar='OUT',
#                     help='path to load data')
parser.add_argument('--train-data-dir', default=train_config['train_dir'], nargs='+', type=str,
                    help='path to load data')
parser.add_argument('--val-data-dir', default=train_config['val_dir'],  nargs='+', type=str,
                    help='path to load data')
parser.add_argument('--num-workers', default=train_config['num_workers'], type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--seed', type=int, help='seed initialization, per-epoch data order and crop augmentation')


def main():
    # Load training configuration
    args = parser.parse_args()
    source_path = args.ckpt or args.init_from
    source = torch.load(source_path, map_location='cpu') if source_path else None
    try:
        training_settings = resolve_training_settings(train_config, source if args.ckpt else None,
            **{key: getattr(args, key) for key in TRAINING_FIELDS})
        for key, value in training_settings.items():
            setattr(args, key, value)
        model_cfg = resolve_model_for_training(net_config, source, args.variant, args.head_activation,
                                               bool(args.ckpt), args.reset_heads)
        model_cfg = configure_training_experiment(model_cfg, args.auxiliary_losses, args.scale_augmentation)
        for name in ('rpn_train_pre_topk', 'rpn_train_post_topk'):
            value = getattr(args, name)
            if value is not None:
                if value < 0:
                    raise ValueError(name + ' must be nonnegative')
                model_cfg[name] = value
    except ValueError as exc:
        parser.error(str(exc))
    if args.seed is None and args.ckpt:
        args.seed = source.get('training_settings', {}).get('seed')
    if args.seed is not None:
        if not 0 <= args.seed < 2**32:
            parser.error('--seed must be in [0, 2**32)')
        training_settings['seed'] = args.seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    start_epoch = int(source['epoch']) + 1 if args.ckpt else 1
    last_epoch = min(args.epochs, args.stop_after_epoch) if args.stop_after_epoch is not None else args.epochs
    if last_epoch < start_epoch:
        parser.error('stop/total epoch must be at least the next training epoch: %d' % start_epoch)
    if not args.ckpt:
        model_dir = Path(args.out_dir) / 'model'
        if (args.init_from and Path(args.out_dir).resolve() == Path(args.init_from).resolve().parent.parent) or any(model_dir.glob('*.ckpt')):
            parser.error('New training requires an output directory without existing checkpoints; use --ckpt to resume')
    data_cfg = dict(config, **model_cfg)
    data_cfg['batch_size'] = args.batch_size
    data_cfg['crop_seed'] = args.seed
    if not (len(args.train_set) == len(args.train_data_dir) == len(args.train_anno)):
        parser.error('--train-set, --train-data-dir and --train-anno must have equal lengths')
    if not (len(args.val_set) == len(args.val_data_dir) == len(args.val_anno)):
        parser.error('--val-set, --val-data-dir and --val-anno must have equal lengths')

    net = args.net
    initial_checkpoint = args.ckpt
    out_dir = args.out_dir
    weight_decay = args.weight_decay
    momentum = args.momentum
    optimizer = args.optimizer
    init_lr = args.init_lr
    epochs = args.epochs
    epoch_save = args.epoch_save
    epoch_rcnn = args.epoch_rcnn
    batch_size = args.batch_size
    lr_schdule = train_config['lr_schedule']
    train_data_dir = args.train_data_dir
    val_data_dir = args.val_data_dir
    train_set_list = args.train_set
    val_set_list = args.val_set
    train_dataset_list = []
    val_dataset_list = []

    #oversample
    max_num = 0
    for item in args.train_anno:
        annos_all = pd.read_csv(item)
        if annos_all.shape[0]>max_num:
            max_num = annos_all.shape[0]
    
    #ensure the alignment of data_source and filelist
    for index in range(len(train_set_list)):
        train_dataset = BboxReader(train_data_dir[index], args.train_set[index], args.train_anno[index], data_cfg, max_num, mode='train')
        train_dataset_list.append(train_dataset)
    for index in range(len(val_set_list)):
        val_dataset = BboxReader(val_data_dir[index], args.val_set[index], args.val_anno[index], data_cfg, max_num, mode='val')
        val_dataset_list.append(val_dataset)

    total_train_dataset = StackDataset(*train_dataset_list)
        
    total_val_dataset = ConcatDataset(val_dataset_list)
    train_order = torch.Generator() if args.seed is not None else None
    val_order = torch.Generator().manual_seed(args.seed) if args.seed is not None else None
    train_loader = DataLoader(total_train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=train_collate,
                              generator=train_order)
    val_loader = DataLoader(total_val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, collate_fn=eval_collate,
                            generator=val_order)

#     train_dataset = BboxReader(train_data_dir, args.train_set, config, mode='train')
#     val_dataset = BboxReader(val_data_dir, args.val_set, config, mode='val')
        
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
#                               num_workers=args.num_workers, pin_memory=True, collate_fn=train_collate)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
#                               num_workers=args.num_workers, pin_memory=True, collate_fn=train_collate)
    
    
    # Initilize network
    net = getattr(this_module, net)(model_cfg)
    if source is not None:
        load_training_weights(net, source, model_cfg, reset_heads=args.reset_heads)
        print('[%s from %s; reset_heads=%s]' % ('Resume' if args.ckpt else 'Weights-only initialization',
                                               source_path, args.reset_heads))
    net = net.cuda()
    
    optimizer = getattr(torch.optim, optimizer)
    optimizer = optimizer(net.parameters(), lr=init_lr, weight_decay=weight_decay, momentum=momentum)

    if initial_checkpoint:
        optimizer.load_state_dict(source['optimizer'])
    del source
    
    model_out_dir = os.path.join(out_dir, 'model')
    tb_out_dir = os.path.join(out_dir, 'runs')
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
    logfile = os.path.join(out_dir, 'log_train')
    sys.stdout = Logger(logfile)
    pyfiles = [f for f in os.listdir('./') if f.endswith('.py') or f.endswith('.sh')]
    for f in pyfiles:
        shutil.copy(f,os.path.join(out_dir,f))

    print('[Training configuration]')
    for arg in vars(args):
        print(arg, getattr(args, arg))

    print('[Model configuration]')
    pprint.pprint(model_cfg)

    print('[start_epoch %d, out_dir %s]' % (start_epoch, out_dir))
    print('[length of train loader %d, length of valid loader %d]' % (len(train_loader), len(val_loader)))

    # Write graph to tensorboard for visualization
    writer = SummaryWriter(tb_out_dir)
    train_writer = SummaryWriter(os.path.join(tb_out_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(tb_out_dir, 'val'))
    # writer.add_graph(net, (torch.zeros((16, 1, 128, 128, 128)).cuda(), [[]], [[]], [[]], [torch.zeros((16, 128, 128, 128))]), verbose=False)

    for i in tqdm(range(start_epoch, last_epoch + 1), desc='Total'):
        if train_order is not None:
            train_order.manual_seed(args.seed + i)
            for dataset in train_dataset_list:
                dataset.epoch = i
        # learning rate schedule
        if isinstance(optimizer, torch.optim.SGD):
            lr = lr_schdule(i, init_lr=init_lr, total=epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = np.inf

        if i >= epoch_rcnn:
            net.use_rcnn = True
        else:
            net.use_rcnn = False

        print('[epoch %d, lr %f, use_rcnn: %r]' % (i, lr, net.use_rcnn))
        train(net, train_loader, optimizer, i, train_writer,
              max_grad_norm=args.max_grad_norm, report_head_health=args.report_head_health)
        validate(net, val_loader, i, val_writer)


        state_dict = net.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        if i % epoch_save == 0 or i == last_epoch:
            torch.save({
                'epoch': i,
                'out_dir': out_dir,
                'model_config': model_cfg,
                'training_settings': training_settings,
                'state_dict': state_dict,
                'optimizer' : optimizer.state_dict()},
                os.path.join(model_out_dir, '%03d.ckpt' % i))

    writer.close()
    train_writer.close()
    val_writer.close()


def train(net, train_loader, optimizer, epoch, writer, max_grad_norm=0., report_head_health=False):
    net.set_mode('train')
    s = time.time()
    rpn_cls_loss, rpn_reg_loss = [], []
    rcnn_cls_loss, rcnn_reg_loss = [], []
    total_loss = []
    rpn_stats = []
    rcnn_stats = []

    for j, (input, truth_box, truth_label) in tqdm(enumerate(train_loader), total=len(train_loader), desc='Train %d' % epoch):
        if len(truth_label)==0:
            continue
        input = Variable(input).cuda()
        try:
            truth_box = np.array(truth_box,dtype=object)
            truth_label = np.array(truth_label,dtype=object)
        except:
            import pdb;pdb.set_trace()

        health = {}
        report_this_batch = report_head_health and j % 200 == 0
        handles = install_health_hooks(net, health) if report_this_batch else []
        try:
            net(input, truth_box, truth_label)
        finally:
            for handle in handles:
                handle.remove()

        loss, rpn_stat, rcnn_stat = net.loss()
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss before optimizer step at epoch %d batch %d' % (epoch, j))
        optimizer.zero_grad()
        loss.backward()
        if report_this_batch:
            for name, parameter in net.named_parameters():
                if name in ('rpn.conv.0.weight', 'rcnn_head.fc1.weight', 'rcnn_head.fc2.weight'):
                    health[name + '_grad_norm'] = float(parameter.grad.norm()) if parameter.grad is not None else None
            probs = net.rpn_logits_flat.detach().sigmoid().reshape(-1)
            labels_flat = net.rpn_labels.reshape(-1)
            weights_flat = net.rpn_label_weights.reshape(-1)
            for label, name in ((1, 'positive'), (0, 'sampled_negative')):
                values = probs[(labels_flat == label) & (weights_flat > 0)]
                health['rpn_' + name + '_mean_probability'] = float(values.mean()) if values.numel() else None
            negatives = probs[(labels_flat == 0) & (weights_flat > 0)]
            positives_count = int((labels_flat == 1).sum())
            k = min(negatives.numel(), net.cfg['num_hard'] * positives_count)
            hard = negatives.topk(k).values if net.cfg['num_hard'] > 0 else negatives
            health['rpn_hard_negative_mean_probability'] = float(hard.mean()) if hard.numel() else None
            health['rpn_hard_negative_max_probability'] = float(hard.max()) if hard.numel() else None
            if net.use_rcnn:
                health['rcnn_positive_samples'] = int((net.rcnn_labels > 0).sum())
                health['rcnn_negative_samples'] = int((net.rcnn_labels == 0).sum())
                health['rcnn_regression_target_absmax'] = float(net.rcnn_targets.abs().max()) if net.rcnn_targets.numel() else None
        if max_grad_norm > 0:
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm, error_if_nonfinite=True)
            if report_this_batch:
                health['total_grad_norm_before_clip'] = float(norm)
        if report_this_batch:
            print('[head_health epoch=%d batch=%d] %s' % (epoch, j, json.dumps(health)))
        optimizer.step()
        # optimizer.zero_grad()
        # with torch.cuda.amp.autocast():
        #     net(input, truth_box, truth_label)
        #     loss, rpn_stat, rcnn_stat = net.loss()
            
        # scaler.scale(loss).backward()

        # # Unscales gradients and calls
        # # or skips optimizer.step()
        # scaler.step(optimizer)

        # # Updates the scale for next iteration
        # scaler.update()

        assert not net.rpn_cls_loss.isnan().any()
        assert not net.rpn_reg_loss.isnan().any()

        rpn_cls_loss.append(net.rpn_cls_loss.item())
        rpn_reg_loss.append(net.rpn_reg_loss.item())
        rcnn_cls_loss.append(net.rcnn_cls_loss.item())
        rcnn_reg_loss.append(net.rcnn_reg_loss.item())
#         import pdb;pdb.set_trace()

        total_loss.append(loss.item())
        rpn_stats.append(rpn_stat)
        rcnn_stats.append(rcnn_stat)

        del input, truth_box, truth_label
        del net.rpn_proposals, net.detections
        del net.total_loss, net.rpn_cls_loss, net.rpn_reg_loss, net.rcnn_cls_loss, net.rcnn_reg_loss
        del net.rpn_logits_flat, net.rpn_deltas_flat

        if net.use_rcnn:
            del net.rcnn_logits, net.rcnn_deltas

    torch.cuda.empty_cache()

    rpn_stats = np.asarray(rpn_stats, np.float32)
    
    print('Train Epoch %d, iter %d, total time %f, loss %f' % (epoch, j, time.time() - s, np.average(total_loss)))
    print('rpn_cls %f, rpn_reg %f, rcnn_cls %f, rcnn_reg %f' % \
        (np.average(rpn_cls_loss), np.average(rpn_reg_loss),
            np.average(rcnn_cls_loss), np.average(rcnn_reg_loss)
            ))
    print('rpn_stats: tpr %f, tnr %f, total pos %d, total neg %d, reg %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
        100.0 * np.sum(rpn_stats[:, 0]) / np.sum(rpn_stats[:, 1]),
        100.0 * np.sum(rpn_stats[:, 2]) / np.sum(rpn_stats[:, 3]),
        np.sum(rpn_stats[:, 1]),
        np.sum(rpn_stats[:, 3]),
        np.mean(rpn_stats[:, 4]),
        np.mean(rpn_stats[:, 5]),
        np.mean(rpn_stats[:, 6]),
        np.mean(rpn_stats[:, 7]),
        np.mean(rpn_stats[:, 8]),
        np.mean(rpn_stats[:, 9])))

    # Write to tensorboard
    writer.add_scalar('loss', np.average(total_loss), epoch)
    writer.add_scalar('rpn_cls', np.average(rpn_cls_loss), epoch)
    writer.add_scalar('rpn_reg', np.average(rpn_reg_loss), epoch)
    writer.add_scalar('rcnn_cls', np.average(rcnn_cls_loss), epoch)
    writer.add_scalar('rcnn_reg', np.average(rcnn_reg_loss), epoch)

    writer.add_scalar('rpn_reg_z', np.mean(rpn_stats[:, 4]), epoch)
    writer.add_scalar('rpn_reg_y', np.mean(rpn_stats[:, 5]), epoch)
    writer.add_scalar('rpn_reg_x', np.mean(rpn_stats[:, 6]), epoch)
    writer.add_scalar('rpn_reg_d', np.mean(rpn_stats[:, 7]), epoch)
    writer.add_scalar('rpn_reg_h', np.mean(rpn_stats[:, 8]), epoch)
    writer.add_scalar('rpn_reg_w', np.mean(rpn_stats[:, 9]), epoch)

    if net.use_rcnn:
        confusion_matrix = np.asarray([stat[-1] for stat in rcnn_stats], np.int32)
        rcnn_stats = np.asarray([stat[:-1] for stat in rcnn_stats], np.float32)
        
        confusion_matrix = np.sum(confusion_matrix, 0)

        print('rcnn_stats: reg %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
            np.mean(rcnn_stats[:, 0]),
            np.mean(rcnn_stats[:, 1]),
            np.mean(rcnn_stats[:, 2]),
            np.mean(rcnn_stats[:, 3]),
            np.mean(rcnn_stats[:, 4]),
            np.mean(rcnn_stats[:, 5])))
        # print_confusion_matrix(confusion_matrix)
        writer.add_scalar('rcnn_reg_z', np.mean(rcnn_stats[:, 0]), epoch)
        writer.add_scalar('rcnn_reg_y', np.mean(rcnn_stats[:, 1]), epoch)
        writer.add_scalar('rcnn_reg_x', np.mean(rcnn_stats[:, 2]), epoch)
        writer.add_scalar('rcnn_reg_d', np.mean(rcnn_stats[:, 3]), epoch)
        writer.add_scalar('rcnn_reg_h', np.mean(rcnn_stats[:, 4]), epoch)
        writer.add_scalar('rcnn_reg_w', np.mean(rcnn_stats[:, 5]), epoch)
    

def validate(net, val_loader, epoch, writer):
    net.set_mode('valid')
    rpn_cls_loss, rpn_reg_loss = [], []
    rcnn_cls_loss, rcnn_reg_loss = [], []
    total_loss = []
    rpn_stats = []
    rcnn_stats = []

    s = time.time()
    for j, (input, truth_box, truth_label) in tqdm(enumerate(val_loader), total=len(val_loader), desc='Val %d' % epoch):
        if len(truth_label)==0:
            continue
        with torch.no_grad():
            input = Variable(input).cuda()
            truth_box = np.array(truth_box,dtype=object)
            truth_label = np.array(truth_label,dtype=object)

            net(input, truth_box, truth_label)
            loss, rpn_stat, rcnn_stat = net.loss()

        rpn_cls_loss.append(net.rpn_cls_loss.item())
        rpn_reg_loss.append(net.rpn_reg_loss.item())
        rcnn_cls_loss.append(net.rcnn_cls_loss.item())
        rcnn_reg_loss.append(net.rcnn_reg_loss.item())

        total_loss.append(loss.item())
        rpn_stats.append(rpn_stat)
        rcnn_stats.append(rcnn_stat)

    rpn_stats = np.asarray(rpn_stats, np.float32)
    print('Val Epoch %d, iter %d, total time %f, loss %f' % (epoch, j, time.time()-s, np.average(total_loss)))
    print('rpn_cls %f, rpn_reg %f, rcnn_cls %f, rcnn_reg %f' % \
        (np.average(rpn_cls_loss), np.average(rpn_reg_loss),
            np.average(rcnn_cls_loss), np.average(rcnn_reg_loss)
            ))
    print('rpn_stats: tpr %f, tnr %f, total pos %d, total neg %d, reg %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
        100.0 * np.sum(rpn_stats[:, 0]) / np.sum(rpn_stats[:, 1]),
        100.0 * np.sum(rpn_stats[:, 2]) / np.sum(rpn_stats[:, 3]),
        np.sum(rpn_stats[:, 1]),
        np.sum(rpn_stats[:, 3]),
        np.mean(rpn_stats[:, 4]),
        np.mean(rpn_stats[:, 5]),
        np.mean(rpn_stats[:, 6]),
        np.mean(rpn_stats[:, 7]),
        np.mean(rpn_stats[:, 8]),
        np.mean(rpn_stats[:, 9])))
    
    # Write to tensorboard
    writer.add_scalar('loss', np.average(total_loss), epoch)
    writer.add_scalar('rpn_cls', np.average(rpn_cls_loss), epoch)
    writer.add_scalar('rpn_reg', np.average(rpn_reg_loss), epoch)
    writer.add_scalar('rcnn_cls', np.average(rcnn_cls_loss), epoch)
    writer.add_scalar('rcnn_reg', np.average(rcnn_reg_loss), epoch)

    writer.add_scalar('rpn_reg_z', np.mean(rpn_stats[:, 4]), epoch)
    writer.add_scalar('rpn_reg_y', np.mean(rpn_stats[:, 5]), epoch)
    writer.add_scalar('rpn_reg_x', np.mean(rpn_stats[:, 6]), epoch)
    writer.add_scalar('rpn_reg_d', np.mean(rpn_stats[:, 7]), epoch)
    writer.add_scalar('rpn_reg_h', np.mean(rpn_stats[:, 8]), epoch)
    writer.add_scalar('rpn_reg_w', np.mean(rpn_stats[:, 9]), epoch)

    if net.use_rcnn:
        confusion_matrix = np.asarray([stat[-1] for stat in rcnn_stats], np.int32)
        rcnn_stats = np.asarray([stat[:-1] for stat in rcnn_stats], np.float32)
        
        confusion_matrix = np.sum(confusion_matrix, 0)
        print('rcnn_stats: reg %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
            np.mean(rcnn_stats[:, 0]),
            np.mean(rcnn_stats[:, 1]),
            np.mean(rcnn_stats[:, 2]),
            np.mean(rcnn_stats[:, 3]),
            np.mean(rcnn_stats[:, 4]),
            np.mean(rcnn_stats[:, 5])))
        # print_confusion_matrix(confusion_matrix)
        writer.add_scalar('rcnn_reg_z', np.mean(rcnn_stats[:, 0]), epoch)
        writer.add_scalar('rcnn_reg_y', np.mean(rcnn_stats[:, 1]), epoch)
        writer.add_scalar('rcnn_reg_x', np.mean(rcnn_stats[:, 2]), epoch)
        writer.add_scalar('rcnn_reg_d', np.mean(rcnn_stats[:, 3]), epoch)
        writer.add_scalar('rcnn_reg_h', np.mean(rcnn_stats[:, 4]), epoch)
        writer.add_scalar('rcnn_reg_w', np.mean(rcnn_stats[:, 5]), epoch)
    
    del input, truth_box, truth_label
    del net.rpn_proposals, net.detections
    del net.total_loss, net.rpn_cls_loss, net.rpn_reg_loss, net.rcnn_cls_loss, net.rcnn_reg_loss

    if net.use_rcnn:
        del net.rcnn_logits, net.rcnn_deltas

    torch.cuda.empty_cache()

def print_confusion_matrix(confusion_matrix):
    line_new = '{:>4}  ' * (len(config['roi_names']) + 2)
    print(line_new.format('gt/p', *list(range(len(config['roi_names']) + 1))))

    for i in range(len(config['roi_names']) + 1):
        print(line_new.format(i, *list(confusion_matrix[i])))
        

if __name__ == '__main__':
    main()
