import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F

class BinActive(torch.autograd.Function):
    ''' Binarize the input activations and calculate the mean across channel dimension. '''
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        output = input.sign()
        return output
    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.gt(1)] = 0
        grad_input[input.lt(-1)] = 0
        return grad_input

binactive = BinActive.apply
class BinLinear(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(BinLinear, self).__init__()
        self.bn = nn.LayerNorm(input_dimension)
        self.weight = nn.Parameter(torch.rand(output_dimension, input_dimension) * 0.001, requires_grad=True)
        self.bias = nn.Parameter(torch.rand(output_dimension) * 0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(output_dimension))

    def forward(self, x):
        x = self.bn(x)
        x = binactive(x)
        real_weight = self.weight
        mean_weights = real_weight.mul(-1).mean(dim=1, keepdim=True).expand_as(self.weight).contiguous()
        centered_weights = real_weight.add(mean_weights)
        cliped_weights = torch.clamp(centered_weights, -1.0, 1.0)
        signed_weights = torch.sign(centered_weights).detach() - cliped_weights.detach() + cliped_weights
        x = F.linear(x, signed_weights, self.bias)
        return x.mul(self.alpha)



