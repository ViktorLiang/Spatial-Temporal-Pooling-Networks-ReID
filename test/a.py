import random as rand

import torch
import numpy as np


class MetrixMultiply(Function):
    @staticmethod
    def forward(ctx, input1, input2, weight):
        ctx.save_for_backward(weight)
        output = input1.mm(weight)
        output = output.mm(input2.t())
        tanh = nn.Tanh()
        return tanh(output)

    @staticmethod
    def backward(ctx, grad_outputs):
        print "---MetrixMultiply---backward---"
        weight, = ctx.saved_variables
        grad_input = grad_outputs * 1
        # print grad_outputs.shape, weight.shape
        grad_weight = grad_outputs.mm(weight)
        return grad_input, grad_input, grad_weight


# input1 = torch.randn((4, 6), requires_grad=False)
# input2 = torch.randn((4, 6), requires_grad=False)
# weight = torch.randn((6, 6), requires_grad=True)
# met = MetrixMultiply.apply(input1, input2, weight)
# grad_input = torch.zeros((6, 6), requires_grad=True)
# grad_input[0][0] = 1
# met.backward(grad_input)

# a = {}
# for i in range(10):
#     a[i] = {}
#     a[i][0] = torch.randn(5,54,47).type(torch.cuda.FloatTensor)
#     a[i][1] = torch.randn(5,54,47).type(torch.cuda.FloatTensor)


data = [[[], []], [[], []]]
data[0][1] = torch.randn(30, 5, 64, 48)
data[0][0] = torch.randn(30, 5, 64, 48)
data[1][0] = torch.randn(30, 5, 64, 48)
data[1][1] = torch.randn(40, 5, 64, 48)
sp = data[0][0].shape

len = 20
idx = rand.randint(0, sp[0] - len - 1) + 1
print idx, '--', idx + len
data[0][0][idx:idx + len]