# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np


class AttenMetrix(nn.Module):
    def __init__(self, embedding_size):
        super(AttenMetrix, self).__init__()
        self.atten_weight = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.Tensor(embedding_size, embedding_size).type(
                    torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
            requires_grad=True)

    def forward(self, seq1, seq2):
        o1 = seq1.mm(self.atten_weight)
        o2 = o1.mm(seq2.t())
        tanh = nn.Tanh()
        return tanh(o2)


class Attention(nn.Module):
    def __init__(self, embedding_size):
        super(Attention, self).__init__()
        self.embedding_size = embedding_size
        # self.n_person_train = n_person_train

        self.atten_metrix = AttenMetrix(embedding_size)
        self.add_module('atten_metrix', self.atten_metrix)

    def attent(self, seq_p, seq_g):
        atten_weight = torch.rand((self.embedding_size, self.embedding_size), requires_grad=True)
        atten_output = self.metrix.apply(seq_p, seq_g, atten_weight)
        return atten_output

    def build_net(self, seq_p, seq_g):
        # atten_output的注意力机制将seq_p和seq_g的图像融合在一起, shape:embedding_size*embedding_size
        atten_output = self.atten_metrix(seq_p, seq_g)
        # 分别提取注意力分部的行和列的max值,行的对应seq_p,列的对应seq_g, shape:seq_length*1
        tp, _ = torch.max(atten_output, 1)
        tg, _ = torch.max(atten_output, 0)
        # 计算注意力概率
        softmax = nn.Softmax(dim=0)
        tp_s = softmax(tp)
        tg_s = softmax(tg)
        tp_s = torch.unsqueeze(tp_s, 0)
        tg_s = torch.unsqueeze(tg_s, 0)
        # seq矩阵乘对应概率得到注意力机制对应的seq局部, shape: 1*embedding_size
        v_p = tp_s.mm(seq_p)
        v_g = tg_s.mm(seq_g)

        return v_p, v_g

    def forward(self, input1, input2):
        return self.build_net(input1, input2)
