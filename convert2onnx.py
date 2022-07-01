
from thop import profile
import torch

from model import UNet

x = torch.randn(1, 3, 384, 384, requires_grad=True)

model = UNet(3, 3, 16, [])


torch.onnx.export(model, x, "helloworld/unet.onnx",
                  export_params=True,
                  opset_version=10,
                  do_constant_folding=True,
                  input_names = ['input'],
                  output_names = ['output'])



flops, params = profile(model, inputs=(x, ))

print('params:')
print(params / 1e6)
print('flops:')
print(flops / 1e9)
