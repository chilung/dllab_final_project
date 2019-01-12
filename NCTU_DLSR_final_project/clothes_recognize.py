# TESTDATADIR="/tmp/dataset-nctu/clothes/clothes_test/" python3 detection.py

from benchmark import benchmarking
import sys
import os
import glob

import torch
import torch.nn as nn
from torchvision import transforms

module_path = os.path.abspath(os.path.join('clothes'))
sys.path.append(module_path)

import dataset
from utils import *
from image import correct_yolo_boxes
from cfg import parse_cfg
from darknet import Darknet


def list_file(dataset_dir):
    filenames = glob.glob('{}/images/*.jpg'.format(dataset_dir))
    
    list_filenames = open('clothes/cfg/valid.txt', 'w')
    for filename in filenames:
        list_filenames.write('%s\n'%(filename))
    list_filenames.close()


weightfile = 'clothes/cfg/000200e.weights'
#weightfile = 'clothes/cfg/000080d.weights'
cfgfile = 'clothes/cfg/clothes.cfg'
datacfg = 'clothes/cfg/clothes.data'

# global variables
use_cuda      = True
eps           = 1e-5
conf_thresh   = 0.25
nms_thresh    = 0.4
iou_thresh    = 0.5
shape         = (416, 416)

data_options  = read_data_cfg(datacfg)
net_options   = parse_cfg(cfgfile)[0]
try:
    data_dir = os.environ['TESTDATADIR']
    list_file(data_dir)
except KeyError:
    pass
testlist = data_options['valid']
gpus          = data_options['gpus']
ngpus         = len(gpus.split(','))
num_workers   = int(data_options['num_workers'])
batch_size    = int(net_options['batch'])
device        = "cuda" if use_cuda else "cpu"
    

def load_testlist(testlist):
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
    loader = torch.utils.data.DataLoader(
        dataset.listDataset(testlist, shape=(416, 416),
            shuffle=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                ]), train=False),
        batch_size=batch_size, shuffle=False, **kwargs)
    return loader


model = Darknet(cfgfile, use_cuda=use_cuda)
model.load_weights(weightfile)
if use_cuda and ngpus > 1:
    model = torch.nn.DataParallel(model)

def truths_length(truths):
    for i in range(50):
        if truths[i][1] == 0:
            return i
    return 50
    
@benchmarking(team=3, task=2, model=model, preprocess_fn=None)
def inference(model, test_loader, **kwargs):
    beta = kwargs['beta']
    #device = kwargs['device']
    beta_s = beta * beta
    
    model = model.to(device)
    model.eval()
    
    total       = 0.0
    proposals   = 0.0
    correct     = 0.0
        
    for batch_idx, (data, target, org_w, org_h) in enumerate(test_loader):
        data = data.to(device)
        output = model(data)
        all_boxes = get_all_boxes(output, shape, conf_thresh, 80, use_cuda=use_cuda)

        for k in range(len(all_boxes)):
            boxes = all_boxes[k]
            correct_yolo_boxes(boxes, org_w[k], org_h[k], 416, 416)
            boxes = np.array(nms(boxes, nms_thresh))
            num_pred = len(boxes)
            if num_pred == 0:
                continue
                
            truths = target[k].view(-1, 5)
            num_gts = truths_length(truths)
            total = total + num_gts
            
            pred_boxes = torch.FloatTensor(boxes).t()
            proposals += int((boxes[:,4]>conf_thresh).sum())
            for i in range(num_gts):
                gt_boxes = torch.FloatTensor([truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]])
                gt_boxes = gt_boxes.repeat(num_pred,1).t()
                best_iou, best_j = torch.max(multi_bbox_ious(gt_boxes, pred_boxes, x1y1x2y2=False),0)
                if best_iou > iou_thresh and pred_boxes[6][best_j] == gt_boxes[6][0]:
                    correct += 1
        
        #prec_i = 1.0*correct/(proposals+eps)
        #reca_i = 1.0*correct/(total+eps)
        #fsco_i = (1.0+beta_s)*prec_i*reca_i/(beta_s*prec_i+reca_i+eps)
        #print('%03d-th test, correct: %03d, precision: %f, recall: %f, fscore: %f' % (batch_idx, correct, prec_i, reca_i, fsco_i))
                        
    precision = 1.0*correct/(proposals+eps)
    recall = 1.0*correct/(total+eps)
    fscore = (1.0+beta_s)*precision*recall/(beta_s*precision+recall+eps)
    print("Final correct: %03d, precision: %f, recall: %f, fscore: %f" % (correct, precision, recall, fscore))
    return fscore


if __name__=='__main__':
    test_loader = load_testlist(testlist)
    kwargs = {'beta': 1.0}
    inference(model, test_loader, **kwargs)
