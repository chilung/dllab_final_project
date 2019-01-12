import torch

dummy_input = torch.randn(1, 3, 4, 5, device='cpu')
print(list(dummy_input.size())[0])
print(list(dummy_input.size())[1])
print(list(dummy_input.size())[2])
print(list(dummy_input.size())[3])

kk = torch.randn(list(dummy_input.size())[0],
        list(dummy_input.size())[1],
        list(dummy_input.size())[2],
        list(dummy_input.size())[3],
        device='cuda', requires_grad=False)
print(kk.size())
