import os
import math
from decimal import Decimal

import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm
from collections import OrderedDict

dim0 = 0
dim2 = 0
dim3 = 0

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        global dim0, dim2, dim3
        
        self.optimizer.schedule()
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            
            dim0 = lr.size()[0] 
            dim2 = lr.size()[2]
            dim3 = lr.size()[3]

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, idx_scale)
            loss = self.loss(sr, hr)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        import onnx
        from ngraph_onnx.onnx_importer.importer import import_onnx_model
        import ngraph as ng
        global dim0, dim2, dim3

        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch() + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        # print(self.loader_test)
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                print('idx_scale={}'.format(idx_scale))
                # print("len: {}".format(len(d)))
                # for lr, hr, filename, _ in tqdm(d, ncols=80):
                for batch, (lr, hr, filename, _) in enumerate(d):
                    print('{} '.format(batch), end='', flush=True)
                    lr, hr = self.prepare(lr, hr)
                    print('test lr.size: {}'.format(lr.size()))
                    dim0 = lr.size()[0]
                    dim2 = lr.size()[2]
                    dim3 = lr.size()[3]
                    
                    showbug = False
                    if showbug: print('stage1', flush=True)
                    if self.args.ngraph:
                        
                        pytorch_model_name = self.args.ngraph
                        pytorch_edsr_model = torch.load(pytorch_model_name).cuda()
                        if showbug: print('stage2-1', flush=True)
                        # print(lr.size())
                        # dummy_input = torch.randn_like(lr, device='cuda')
                        if showbug: print('stage2-2', flush=True)
                        edsr_onnx_filename = '{}.onnx'.format(pytorch_model_name)
                        # print('Export to onnx model {}'.format(edsr_onnx_filename))
                        torch.onnx.export(pytorch_edsr_model, lr.to(torch.device('cuda')), edsr_onnx_filename, export_params=True, verbose=False, training=False)
                        if showbug: print('stage2-3', flush=True)

                        edsr_onnx_model = onnx.load(edsr_onnx_filename)
                        # print(onnx.helper.printable_graph(edsr_onnx_model.graph))

                        if showbug: print('stage2-4', flush=True)
                        ng_models = import_onnx_model(edsr_onnx_model)

                        # print('Convert to nGreph Model')

                        ng_model = ng_models[0]
                        if showbug: print('stage2-5', flush=True)
                        runtime = ng.runtime(backend_name='CPU')
                        if showbug: print('stage2-6', flush=True)
                        edsr_ng_model = runtime.computation(ng_model['output'], *ng_model['inputs'])
                        if showbug: print('stage2-7', flush=True)

                        sr = edsr_ng_model(lr, idx_scale)
                        if showbug: print('stage2-8', flush=True)
                        sr = torch.from_numpy(sr)
                        if showbug: print('stage2-9', flush=True)
                    elif self.args.tensorrt:
                        pytorch_model_name = self.args.tensorrt
                        pytorch_edsr_model = torch.load(pytorch_model_name)
                        
                        # lr_np = lr.numpy().astype(np.float32)
                        dummy_input = torch.randn_like(lr, device='cuda')
                        edsr_onnx_filename = '{}.onnx'.format(pytorch_model_name)
                        print('Export to onnx model {}'.format(edsr_onnx_filename))
                        torch.onnx.export(pytorch_edsr_model, dummy_input, edsr_onnx_filename, export_params=True, verbose=False, training=False)

                        import os
                        import onnx

                        edsr_onnx_model = onnx.load(edsr_onnx_filename)
                        # print(onnx.helper.printable_graph(edsr_onnx_model.graph))

                        import tensorrt
                        import onnx_tensorrt.backend as backend
                        import numpy as np

                        tensorrt_engine = backend.prepare(edsr_onnx_model, device='CUDA:0')
                        # lr_np = lr_np.to(torch.device("cuda:0"))
                        # lr.numpy().astype(np.float32)

                        sr = tensorrt_engine.run(lr.numpy().astype(np.float32))[0]
                        sr = torch.from_numpy(sr)

                        print('complete one')   



                        pytorch_model_name = self.args.tensorrt
                        pytorch_edsr_model = torch.load(pytorch_model_name)
                        
                        # lr_np = lr.numpy().astype(np.float32)
                        dummy_input = torch.randn_like(lr, device='cuda')
                        edsr_onnx_filename = '{}.onnx'.format(pytorch_model_name)
                        print('Export to onnx model {}'.format(edsr_onnx_filename))
                        torch.onnx.export(pytorch_edsr_model, dummy_input, edsr_onnx_filename, export_params=True, verbose=False, training=False)

                        import os
                        import onnx

                        edsr_onnx_model = onnx.load(edsr_onnx_filename)
                        # print(onnx.helper.printable_graph(edsr_onnx_model.graph))

                        import tensorrt
                        import onnx_tensorrt.backend as backend
                        import numpy as np

                        tensorrt_engine = backend.prepare(edsr_onnx_model, device='CUDA:0')
                        # lr_np = lr_np.to(torch.device("cuda:0"))
                        # lr.numpy().astype(np.float32)

                        sr = tensorrt_engine.run(lr.numpy().astype(np.float32))[0]
                        sr = torch.from_numpy(sr)
                        
                        print('complete two')   
                    else:
                        sr = self.model(lr, idx_scale)

                    if showbug: print('stage3', flush=True)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    if showbug: print('stage4', flush=True)
                    save_list = [sr]
                    if showbug: print('stage5', flush=True)
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    if showbug: print('stage6', flush=True)
                    if self.args.save_gt:
                        save_list.extend([lr, hr])
                    if showbug: print('stage7', flush=True)

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)
                    if showbug: print('stage8', flush=True)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                psnr = self.ckp.log[-1, idx_data, idx_scale].numpy()
                print('')
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )
                
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
           'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)
        return psnr

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

