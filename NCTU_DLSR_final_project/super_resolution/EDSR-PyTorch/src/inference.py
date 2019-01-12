import subprocess
import os
import sys
import re
from collections import OrderedDict
import shlex

def get_psnr_and_time(fname):
    metric = OrderedDict()
    with open(fname) as fp:  
        for cnt, line in enumerate(fp):
            # print("Line {}: {}".format(cnt, line))
            pattern =   re.compile(r"\[(\S+) x2\](\s+)PSNR: (\d+.\d+) \(Best: (\d+.\d+) @epoch 1\)")
            match = pattern.match(line)
            if match:
                test_set_str, white_space_str, psnr_str, best_psnr_str = match.groups()
                # print(test_set_str, 'white space', psnr_str, best_psnr_str)
                      
            # Accuracy of the network on the 3347 test images: 92.41%, and loss is: 0.004
            pattern = re.compile(r"Forward: (\d+.\d+)s")
            match = pattern.match(line)
            if match:
                time_str, = match.groups()
                # print(time_str)

    metric['psnr'] = float("{:.3f}".format(float(psnr_str)))
    metric['time'] = float("{:.2f}".format(float(time_str)))

    return metric

# @benchmarking(team=3, task=1, model=net, preprocess_fn=None)
def inference_fn(**kwargs):
    
    dev = kwargs['device']
    dir_data = kwargs['dir_data']
    dataset = kwargs['dataset']
    model_name = kwargs['model_name']
    
    elif dev == 'cuda':
        with open("gpu.download.log", "w") as log_file:
             subprocess.run(shlex.split('python main.py --dir_data {} --data_test {} --scale 2 --pre_train {} --test_only'.format(dir_data, dataset, model_name)), stdout=log_file)
        log_file.close()
        metric = get_psnr_and_time("gpu.download.log")
    if dev == 'cpu':
        with open("cpu.download.log", "w") as log_file:
             subprocess.run(shlex.split('python main.py --dir_data {} --data_test {} --scale 2 --cpu --ngraph {} --test_only'.format(dir_data, dataset, model_name)), stdout=log_file)
        log_file.close()
        metric = get_psnr_and_time("cpu.download.log")
    if dev == 'cpu-ngraph':
        with open("cpu.download.log", "w") as log_file:
             subprocess.run(shlex.split('python main.py --dir_data {} --data_test {} --scale 2 --cpu --ngraph {} --test_only'.format(dir_data, dataset, model_name)), stdout=log_file)
        log_file.close()
        metric = get_psnr_and_time("cpu.download.log")
        
    metric['device'] = dev
    metric['model size'] = os.path.getsize(model_name)
    return metric

print(inference_fn(device='cuda', dir_data='./', dataset='valid', model_name='./edsr_baseline_download.pt'))
print(inference_fn(device='cpu', dir_data='./', dataset='valid', model_name='./edsr.model'))
