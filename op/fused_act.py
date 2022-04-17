import os

import torch
from torch import nn
from torch.nn import functional as F


module_path = os.path.dirname(__file__)

class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        if bias:
            self.bias = nn.Parameter(torch.zeros(channel))

        else:
            self.bias = None

        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)

def bias_broadcasting(input_shape, bias_shape):
    '''
    In the input size list, find the position of the sublist that matches to the bias size 
    '''
    input_shape = torch.tensor(input_shape)
    bias_shape = torch.tensor(bias_shape)

    input_shape_windows = input_shape.unfold(dimension=0, size=len(bias_shape), step=1)
    pos = (input_shape_windows == bias_shape).nonzero(as_tuple=True)[0][0]
    
    new_bias_view = [1] * len(input_shape)
    new_bias_view[pos : pos + len(bias_shape)] = bias_shape
    return new_bias_view

def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    if input.device.type == "cpu":
        if bias is not None:
            new_bias_view = bias_broadcasting(input.shape, bias.shape)
            return (
                F.leaky_relu(
                    input + bias.view(new_bias_view), negative_slope=0.2
                )
                * scale
            )

        else:
            return F.leaky_relu(input, negative_slope=0.2) * scale

    else:
        return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)
