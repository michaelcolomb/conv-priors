import torch
from torch import nn
import numpy as np


class GPConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, input_size,
                 rank=10, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.input_size = np.array(input_size)
        self.rank = rank
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        self.output_size = np.floor((self.input_size + 2 * self.padding - self.kernel_size) / self.stride + 1).astype(np.int)
        self.m = np.prod(self.output_size)
        
        self.conv = nn.Conv2d(self.in_channels, self.out_channels * self.rank, kernel_size=self.kernel_size, 
                              stride=self.stride, padding=self.padding, bias=self.bias)
        
        self.B = nn.Parameter(2 * torch.rand(1, self.out_channels, self.m, self.rank, 1) - 1)
        
    def forward(self, x):
        D = self.conv(x).view(x.shape[0], self.out_channels, self.rank, 1, -1).transpose(-3, -1)
        output = torch.matmul(D, self.B)
        output = output.view(*output.shape[:2], *self.output_size)
        return output
    
    @property
    def filters(self):
        A = self.conv.weight.view(self.out_channels, self.rank, -1)
        A = A.transpose(-2, -1)
        B = self.B.contiguous().squeeze().transpose(-2, -1)
        return torch.matmul(A, B)
        
        
        
