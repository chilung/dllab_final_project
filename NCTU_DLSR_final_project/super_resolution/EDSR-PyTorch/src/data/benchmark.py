import os

from data import common
from data import srdata

import numpy as np

import torch
import torch.utils.data as data

class Benchmark(srdata.SRData):
    def __init__(self, args, name='', train=True, benchmark=True):
        super(Benchmark, self).__init__(
            args, name=name, train=train, benchmark=True
        )

    def _set_filesystem(self, dir_data):
        # print('in benchmark {}, name: {}'.format(dir_data, self.name))
        if self.name == 'valid':
            self.apath = os.path.join(dir_data, 'DIV2K')
        else:
            self.apath = os.path.join(dir_data, 'benchmark', self.name)
        self.dir_hr = os.path.join(self.apath, 'DIV2K_valid_HR' if self.name == 'valid' else 'HR')
        if self.input_large:
            self.dir_lr = os.path.join(self.apath, 'DIV2K_valid_LR_bicubicL' if self.name == 'valid' else 'LR_bicubicL')
        else:
            self.dir_lr = os.path.join(self.apath, 'DIV2K_valid_LR_bicubic' if self.name == 'valid' else 'LR_bicubic')
        
        assert (os.path.isdir(self.dir_hr)==True and os.path.isdir(self.dir_lr)==True), '\n\n\
ERROR: TEST DATASET not found.\n\
       Make sure your testset are put in directory:\n\
       [TESTDATADIR]/DIV2K/DIV2K_valid_HR\n\
         and \n\
       [TESTDATADIR]/DIV2K/DIV2K_valid_LR_bicubic\n\n'
        
        self.ext = ('', '.png')

