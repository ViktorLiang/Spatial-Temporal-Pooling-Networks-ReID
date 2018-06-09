import torch
import torch.nn as nn


def rnn_t1():
    rnn = nn.RNN(10, 20, 1)

    input = torch.randn(2, 4, 10)
    print (input)
    h0 = torch.randn(1, 4, 20)

    output, hn = rnn(input, h0)
    print output.shape
    print hn.shape


def conv3d_t1():
    m = nn.Conv3d(16, 33, 3, stride=2)
    # non-square kernels and unequal stride and with padding
    m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
    input = torch.randn(20, 16, 10, 50, 100)
    output = m(input)
    print output.shape


def squze():
    input = torch.randn((16,128))
    # m = torch.unsqueeze(input,0)
    input.unsqueeze_(0)
    print input.shape

# conv3d_t1()
# rnn_t1()
squze()