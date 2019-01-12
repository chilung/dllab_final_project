import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
from benchmark import benchmarking

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

model = model.Model(args, checkpoint)
loader = data.Data(args)
loss = loss.Loss(args, checkpoint) if not args.test_only else None

@benchmarking(team=3, task=1, model=model, preprocess_fn=None)
def inference(model, data_loader, **kwargs):
    dev = kwargs['device']
    if dev == 'cuda':
        args.cpu = False
        t = Trainer(args, data_loader, model, loss, checkpoint)
        metric = t.test()
    if dev == 'cpu':
        args.cpu = False
        t = Trainer(args, data_loader, model, loss, checkpoint)
        metric = t.test()
    return metric

if __name__=='__main__':
    inference(model, loader)
