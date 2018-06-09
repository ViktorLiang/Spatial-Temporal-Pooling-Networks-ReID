# -*- encoding: utf-8 -*-
import sys

from attention import *

sys.path.append("/home/ly/workspace/git/Person_Re-id/STPN-ReID-py")
import reidTrain


class Net(nn.Module):
    def __init__(self, args, nFltrs1, nFltrs2, nFltrs3, n_person_train):
        super(Net, self).__init__()
        self.args = args
        self.nFilters = [nFltrs1, nFltrs2, nFltrs3]
        self.n_person_train = n_person_train
        self.input_channels = 5
        self.kernel_size = [5, 5, 5]

        self.poolsize = [2, 2, 2]
        self.stepSize = [2, 2, 2]
        self.padDim = 4

        # conv layers and pooling tanh layers
        self.conv1 = nn.Conv2d(self.input_channels, self.nFilters[0], self.kernel_size[0], stride=1,
                               padding=self.padDim)
        self.conv2 = nn.Conv2d(self.nFilters[0], self.nFilters[1], self.kernel_size[1], stride=1,
                               padding=self.padDim)
        self.conv3 = nn.Conv2d(self.nFilters[1], self.nFilters[2], self.kernel_size[2], stride=1,
                               padding=self.padDim)

        self.pooling1 = nn.MaxPool2d(self.poolsize[0], self.stepSize[0])
        self.pooling2 = nn.MaxPool2d(self.poolsize[1], self.stepSize[1])

        self.tanh = nn.Tanh()

        # spatial pooling layers
        self.pool1 = nn.Sequential(
            nn.AdaptiveMaxPool2d((8, 8)),
        )
        self.pool2 = nn.Sequential(
            nn.AdaptiveMaxPool2d((4, 4)),
        )
        self.pool3 = nn.Sequential(
            nn.AdaptiveMaxPool2d((2, 2)),
        )
        self.pool4 = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
        )

        # full connect layer 1
        n_fully_connected = 32 * (64 + 16 + 4 + 1)
        self.seq2 = nn.Sequential(
            nn.Dropout(self.args.dropoutFrac),
            nn.Linear(n_fully_connected, self.args.embeddingSize),
        )

        # rnn layer
        self.rnn = nn.RNN(self.args.embeddingSize, self.args.embeddingSize)

        # attention layer
        # self.hid_weight = torch.randn((1, self.args.sampleSeqLength, self.args.embeddingSize), requires_grad=True)
        self.hid_weight = nn.Parameter(
            nn.init.xavier_uniform(torch.Tensor(1, self.args.sampleSeqLength, self.args.embeddingSize).type(
                torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
            ), gain=np.sqrt(2.0)), requires_grad=True)

        self.atten = Attention(self.args.embeddingSize)
        self.add_module('atten', self.atten)

        # final full connect layer
        self.final_full_connect = nn.Linear(self.args.embeddingSize, self.n_person_train)

    def build_net(self, input1, input2):
        seq1 = nn.Sequential(
            self.conv1, self.tanh, self.pooling1,
            self.conv2, self.tanh, self.pooling2,
            self.conv3, self.tanh,
        )

        psn1_seq1_out = seq1(input1)
        psn2_seq1_out = seq1(input2)
        psn1_poling = self.spatial_pooling(psn1_seq1_out)
        psn2_poling = self.spatial_pooling(psn2_seq1_out)

        psn1_seq2_out = self.seq2(psn1_poling).unsqueeze_(0)
        psn2_seq2_out = self.seq2(psn2_poling).unsqueeze_(0)

        psn1_rnn_out, hn1 = self.rnn(psn1_seq2_out, self.hid_weight)
        psn2_rnn_out, hn2 = self.rnn(psn2_seq2_out, self.hid_weight)

        psn1_rnn_out_quzed = psn1_rnn_out.squeeze()
        psn2_rnn_out_quzed = psn2_rnn_out.squeeze()
        feature_p, feature_g = self.atten(psn1_rnn_out_quzed, psn2_rnn_out_quzed)
        identity_p = self.final_full_connect(feature_p)
        identity_g = self.final_full_connect(feature_g)
        return feature_p, feature_g, identity_p, identity_g

    def spatial_pooling(self, input):
        out1 = self.pool1(input)
        out1_shape = out1.shape
        sec_dim = out1_shape[-1] * out1_shape[-2] * out1_shape[-3]
        out1 = out1.contiguous().view(-1, sec_dim)

        out2 = self.pool2(input)
        out2_shape = out2.shape
        sec_dim = out2_shape[-1] * out2_shape[-2] * out2_shape[-3]
        out2 = out2.contiguous().view(-1, sec_dim)

        out3 = self.pool3(input)
        out3_shape = out3.shape
        sec_dim = out3_shape[-1] * out3_shape[-2] * out3_shape[-3]
        out3 = out3.contiguous().view(-1, sec_dim)

        out4 = self.pool4(input)
        out4_shape = out4.shape
        sec_dim = out4_shape[-1] * out4_shape[-2] * out4_shape[-3]
        out4 = out4.contiguous().view(-1, sec_dim)

        cat = torch.cat((out1, out2, out3, out4), 1)
        return cat

    def forward(self, input1, input2):
        # (personId,imageId,imgChannels,imgHeight,imgWidth)
        feature_p, feature_g, identity_p, identity_g = self.build_net(input1, input2)
        return feature_p, feature_g, identity_p, identity_g


class Criterion(nn.Module):
    def __init__(self, hinge_margin):
        super(Criterion, self).__init__()
        self.hinge_margin = hinge_margin

    def forward(self, feature_p, feature_g, identity_p, identity_g, target):
        log_soft = nn.LogSoftmax(1)
        lsoft_p = log_soft(identity_p)
        lsoft_g = log_soft(identity_g)
        dist = nn.PairwiseDistance(p=2)
        pair_dist = dist(feature_p, feature_g)

        hing = nn.HingeEmbeddingLoss(margin=self.hinge_margin, reduce=False)
        label0 = torch.tensor(target[0]).type(
            torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor)
        hing_loss = hing(pair_dist, label0)

        nll = nn.NLLLoss()
        label1 = torch.tensor([target[1]]).type(
            torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor)
        label2 = torch.tensor([target[2]]).type(
            torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor)
        lsoft_loss_p = nll(lsoft_p, label1)
        lsoft_loss_g = nll(lsoft_g, label2)

        total_loss = hing_loss + lsoft_loss_p + lsoft_loss_g
        return total_loss


if __name__ == '__main__':
    train = reidTrain.ReidTrain()
    net = Net(train.args, 16, train.args.nConvFilters, train.args.nConvFilters, 2)
    out1, out2 = net.forward()
    print out1.shape
    print out2.shape
