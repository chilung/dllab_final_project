import subprocess
import os
import sys
import re
from collections import OrderedDict
import shlex
import argparse

parser = argparse.ArgumentParser(description='EDSR test',
        usage="\n\
        python {} --device 'cuda' --dataset valid --dir-data './' --model-name './edsr_baseline_download.pt'\n\
        python {} --device 'cpu' --dataset valid --dir-data './' --model-name './edsr_baseline_download.pt'\n\
        python {} --device 'cpu-ngraph' --dataset valid --dir-data './' --model-name 'edsr.model'\n\
        ".format(__file__, __file__, __file__))
parser.add_argument('--device', default='cuda', type=str,
                    help='set the device to cuda, cpu or cpu-ngraph')
parser.add_argument('--dir-data', default='./', type=str,
                    help='set the dataset root directory')
parser.add_argument('--dataset', default='valid', type=str,
                    help='the test dataset, the data should be put into [dir_data]/benchmark/[dataset], and come with HR and LR, make sure the dataset is described in data/__init__.py')
parser.add_argument('--model-name', default='', type=str,
                    help='from the path and filename of the model')

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

    # print(kwargs)
    dev = kwargs['device']
    dir_data = kwargs['dir_data']
    dataset = kwargs['dataset']
    model_name = kwargs['model_name']

    if dev == 'cuda':
        with open("gpu.download.log", "w") as log_file:
             subprocess.run(shlex.split('python main.py --dir_data {} --data_test {} --scale 2 --pre_train {} --test_only'.format(dir_data, dataset, model_name)), stdout=log_file)
        log_file.close()
        metric = get_psnr_and_time("gpu.download.log")
    elif dev == 'cpu':
        with open("cpu.download.log", "w") as log_file:
             subprocess.run(shlex.split('python main.py --dir_data {} --data_test {} --scale 2 --cpu --pre_train {} --test_only'.format(dir_data, dataset, model_name)), stdout=log_file)
        log_file.close()
        metric = get_psnr_and_time("cpu.download.log")
    elif dev == 'cpu-ngraph':
        with open("ngraph.download.log", "w") as log_file:
             subprocess.run(shlex.split('python main.py --dir_data {} --data_test {} --scale 2 --cpu --ngraph {} --test_only'.format(dir_data, dataset, model_name)), stdout=log_file)
        log_file.close()
        metric = get_psnr_and_time("ngraph.download.log")

    metric['device'] = dev
    metric['model size'] = os.path.getsize(model_name)
    return metric

args = parser.parse_args()
# print(args)
print(inference_fn(device=args.device, dir_data=args.dir_data, dataset=args.dataset, model_name=args.model_name))

