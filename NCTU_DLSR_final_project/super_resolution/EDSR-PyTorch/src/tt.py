from torch.autograd import Variable
import torch
import torchvision
import os
import onnx
from ngraph_onnx.onnx_importer.importer import import_onnx_model
import ngraph as ng

torch.set_grad_enabled(False)

# lr = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)
lr = torch.randn(1, 3, 512, 512, requires_grad=False)
pytorch_model_name = './edsr.model'
idx_scale = 0

pytorch_edsr_model = torch.load(pytorch_model_name).cuda()

dummy_input = torch.randn(list(lr.size())[0], list(lr.size())[1], list(lr.size())[2], list(lr.size())[3],
        device='cuda', requires_grad=False)
edsr_onnx_filename = '{}.onnx'.format(pytorch_model_name)

torch.onnx.export(pytorch_edsr_model, dummy_input, edsr_onnx_filename, export_params=True, verbose=True, training=False)
edsr_onnx_model = onnx.load(edsr_onnx_filename)
ng_models = import_onnx_model(edsr_onnx_model)
print(ng_models)

ng_model = ng_models[0]
runtime = ng.runtime(backend_name='CPU')
edsr_ng_model = runtime.computation(ng_model['output'], *ng_model['inputs'])

print(edsr_ng_model)

for i in range(100):
    print(i)

    # sr = edsr_ng_model(lr, idx_scale)
    # lr.to(torch.device('cpu'))
    sr = edsr_ng_model(lr)
    # sr = torch.from_numpy(sr)
