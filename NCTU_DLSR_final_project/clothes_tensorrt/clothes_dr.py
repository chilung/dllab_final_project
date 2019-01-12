# TESTDATADIR="/tmp/dataset-nctu/clothes/clothes_test/" python3 clothes_dr.py

#from benchmark import benchmarking
import torch
import os
import sys
import glob

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
#from PIL import ImageDraw

from data_processing import PreprocessYOLO, PostprocessYOLO
from utils import read_truths_args, multi_bbox_ious

import common
import time
t2 = time.time()
TRT_LOGGER = trt.Logger()

try:
    data_dir = os.environ['TESTDATADIR']
except KeyError:
    data_dir = '/tmp/dataset-nctu/clothes/clothes_test'


def get_engine(engine_file_path="clothes.trt"):
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


input_HW = (416, 416)
output_shapes = [(1, 255, 13, 13), (1, 255, 26, 26), (1, 255, 52, 52)]
preprocessor = PreprocessYOLO(input_HW)
postprocessor_args = {"yolo_masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],                    
                      "yolo_anchors": [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),  
                                       (59, 119), (116, 90), (156, 198), (373, 326)],
                      "obj_threshold": 0.5,                                               
                      "nms_threshold": 0.2,                                               
                      "yolo_input_resolution": input_HW}
postprocessor = PostprocessYOLO(**postprocessor_args)
eps           = 1e-6
beta          = 2.0
beta_s        = beta * beta

def test_loader(dataset_dir=data_dir):
    filenames = glob.glob('{}/images/*.jpg'.format(dataset_dir))
    #filenames = [dataset_dir+'/images/2005.jpg']

    loader = []
    for filename in filenames:
        image_raw, data = preprocessor.process(filename)
        shape_orig_WH = image_raw.size
        labfile = filename.replace('images', 'labels').replace('.jpg', '.txt')
        try:
            target = torch.from_numpy(read_truths_args(labfile, 8.0/416).astype('float32'))
        except Exception:
            target = None#torch.zeros(1,5)
        loader.append([data, target, shape_orig_WH])
    return loader
        
def bbox_ious(pred_boxes, gt_boxes):
    cx1 = pred_boxes[0]
    cy1 = pred_boxes[1]
    cx2 = pred_boxes[0]+pred_boxes[2]
    cy2 = pred_boxes[1]+pred_boxes[3]
 
    gx1 = gt_boxes[0]
    gy1 = gt_boxes[1]
    gx2 = gt_boxes[0]+gt_boxes[2]
    gy2 = gt_boxes[1]+gt_boxes[3]
 
    carea = pred_boxes[2] * pred_boxes[3]
    garea = gt_boxes[2] * gt_boxes[3]
 
    x1 = torch.max(cx1, gx1)
    y1 = torch.max(cy1, gy1)
    x2 = torch.min(cx2, gx2)
    y2 = torch.min(cy2, gy2)
    
    z = torch.FloatTensor(x1.shape).zero_()
    w = torch.max(z, x2 - x1)
    h = torch.max(z, y2 - y1)
    area = w * h
    return area / (carea + garea - area)


#model = resnet18(pretrained=True)
#@benchmarking(team=3, task=2, model=model, preprocess_fn=None)
#def inference(net, data_loader,**kwargs):
#    total = 0
#    correct = 0
#    assert kwargs['device'] != None, 'Device error'
#    device = kwargs['device']
#    model.to(device)
#    with torch.no_grad():
#        for batch_idx, (inputs, targets) in enumerate(testloader):
#            inputs, targets = inputs.to(device), targets.to(device)
#            outputs = model(inputs)
#            _, predicted = outputs.max(1)
#            total += targets.size(0)
#            correct += predicted.eq(targets).sum().item()
#    acc = 100.*correct/total
#    return acc

if __name__=='__main__':
    #transform_test = transforms.Compose([
    #transforms.Resize((224,224),),
    #transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #])
    #testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    #inference(model, testloader)
    loader = test_loader(data_dir)
    
    total       = 0.0
    proposals   = 0.0
    correct     = 0.0
    time_d = 0
    time_1 = 0
    
    with get_engine() as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        
        for batch_idx, (data, target, shape_orig_WH) in enumerate(loader):
            t3 = time.time()
            trt_outputs = []
            inputs[0].host = data
            t0 = time.time()
            trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            t1 = time.time()
            time_d += t1-t0
            trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
            boxes, classes, scores = postprocessor.process(trt_outputs, (shape_orig_WH))
            
            try:
                num_gts = len(target)
                num_pred = len(boxes)
            except:
                continue
            total += num_gts
            proposals += num_pred
            width, height = shape_orig_WH
            boxes /= [width, height, width, height]
            
            pred_boxes = []
            for i in range(len(classes)):
                box = [boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], classes[i]]
                pred_boxes.append(box)
            pred_boxes = torch.FloatTensor(pred_boxes).t()
            
            for i in range(num_gts):
                gt_boxes = torch.FloatTensor([target[i][1], target[i][2], target[i][3], target[i][4], target[i][0]])
                gt_boxes = gt_boxes.repeat(num_pred,1).t()
                ious = bbox_ious(pred_boxes, gt_boxes)
                best_iou, best_j = torch.max(ious,0)
                if pred_boxes[4][best_j] == gt_boxes[4][0]:
                    correct += 1
            #print('total',total)
            #print('proposals',proposals)
            #print('correct',correct)
            #print()
            t4 = time.time()
            time_1 += t4-t3
        
        #prec_i = 1.0*correct/(proposals+eps)
        #reca_i = 1.0*correct/(total+eps)
        #fsco_i = (1.0+beta_s)*prec_i*reca_i/(beta_s*prec_i+reca_i+eps)
        #print('%03d-th test, correct: %03d, precision: %f, recall: %f, fscore: %f' % (batch_idx, correct, prec_i, reca_i, fsco_i))
    precision = 1.0*correct/(proposals+eps)
    recall = 1.0*correct/(total+eps)
    fscore = (1.0+beta_s)*precision*recall/(beta_s*precision+recall+eps)
    print("Final correct: %03d, precision: %f, recall: %f, fscore: %f" % (correct, precision, recall, fscore))
    print('execu time:', time_d)
    print('time_1:', time_1)
    #return fscore