from benchmark import benchmarking
import argparse
import time
import os
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchnet.meter as tnt

module_path = os.path.abspath(os.path.join('cinic10'))
sys.path.append(module_path)
import distiller
import apputils
from models import ALL_MODEL_NAMES, create_model

parser = argparse.ArgumentParser(description='Distiller image classification model compression')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20_cifar',
                    choices=ALL_MODEL_NAMES,
                    help='model architecture: ' +
                    ' | '.join(ALL_MODEL_NAMES) +
                    ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--deterministic', '--det', action='store_true',
                    help='Ensure deterministic execution for re-producible results.')
parser.add_argument('--gpus', metavar='DEV_ID', default=None,
                    help='Comma-separated list of GPU device IDs to be used (default is to use all available devices)')
parser.add_argument('--validation-size', '--vs', type=float, default=0.1,
                    help='Portion of training dataset to set aside for validation')
parser.add_argument('--cpu', default=False,
                    help='use CPU or GPU in evaluation stage')

filename = './cinic10/checkpoint.pth.tar'
setting = ['./cinic10/', '--resume', filename, '--evaluate']
args = parser.parse_args(args=setting)
args.dataset = 'cinic10_npz'
model = create_model(args.pretrained, args.dataset, args.arch, device_ids=args.gpus)
model, _, _ = apputils.load_checkpoint(model, chkpt_file=args.resume)
_, _, test_loader, _ = apputils.load_data(
        args.dataset, os.path.expanduser(args.data), args.batch_size,
        args.workers, args.validation_size, args.deterministic)
    
def do_inference(setting):
    args = parser.parse_args(args=setting)
    
    if args.gpus is not None:
        torch.cuda.set_device(args.gpus[0])

    args.dataset = 'cinic10_npz'
    model = create_model(args.pretrained, args.dataset, args.arch, device_ids=args.gpus)
    
    if args.resume:
        model, _, _ = apputils.load_checkpoint(
            model, chkpt_file=args.resume)

    criterion = nn.CrossEntropyLoss().cuda()    
    
    _, _, test_loader, _ = apputils.load_data(
        args.dataset, os.path.expanduser(args.data), args.batch_size,
        args.workers, args.validation_size, args.deterministic)
    
    return validate(model, criterion, test_loader, args)


def validate(model, criterion, data_loader, args):
    classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))
    model.eval()
    if args.cpu==True:
        model = model.cpu()
    
    for validation_step, (inputs, target) in enumerate(data_loader):
        with torch.no_grad():               
            if args.cpu==True:
                inputs, target = inputs.cpu(), target.cpu()
            else:
                inputs, target = inputs.to('cuda'), target.to('cuda')
            
            output = model(inputs)
            classerr.add(output.data, target)                
    return classerr.value(1)


@benchmarking(team=3, task=0, model=model, preprocess_fn=None)
def go(**kwargs):
    device = kwargs['device']
    if device == 'cuda':
        setting = ['--arch=resnet20_cifar', './cinic10/', '--resume', filename, '--evaluate']
        top1 = do_inference(setting)

    else:
        setting = ['--arch=resnet20_cifar', './cinic10/', '--cpu=True', '--resume', filename, '--evaluate']
        top1 = do_inference(setting)
        
    return top1


if __name__=='__main__':
    go()
    
    
    