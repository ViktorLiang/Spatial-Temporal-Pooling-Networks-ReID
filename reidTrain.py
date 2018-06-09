from __future__ import unicode_literals, print_function, division
import os
import argparse

import datasets
import train
from models.spatial_temporal import *


class ReidTrain:
    def __init__(self):
        parser = argparse.ArgumentParser(description='training parameters')
        parser.add_argument('-nEpochs', default=600, type=int, help='number of training epochs')
        parser.add_argument('-samplingEpochs', default=16, type=int,
                            help='how often to compute the CMC curve - dont compute too much - its slow!')
        parser.add_argument('-dataset', default=1, type=int, help='0- ilids debug, 1 -  ilids, 2 - prid, 3 - mars')
        parser.add_argument('-dirRGB', default='', help='dir path to the sequences of original datasets')
        parser.add_argument('-dirOF', default='', help='dir path to the sequences of optical flow')
        parser.add_argument('-sampleSeqLength', default=16, type=int, help='length of sequence to train network')
        parser.add_argument('-gradClip', default=5, type=int, help='magnitude of clip on the RNN gradient')
        parser.add_argument('-saveFileName', default='basicnet', help='name to save dataset file')
        parser.add_argument('-usePredefinedSplit', default=0, type=int,
                            help='Use predefined test/training split loaded from a file')
        parser.add_argument('-dropoutFrac', default=0.6, type=float, help='fraction of dropout to use between layers')
        parser.add_argument('-dropoutFracRNN', default=0.6, type=float,
                            help='fraction of dropout to use between RNN layers')
        parser.add_argument('-disableOpticalFlow', default=0, type=int, help='use optical flow features or not')
        parser.add_argument('-seed', default=1, type=int, help='random seed')
        parser.add_argument('-learningRate', default=1e-3, type=float)
        parser.add_argument('-momentum', default=0.9, type=float)
        parser.add_argument('-nConvFilters', default=32, type=int)
        parser.add_argument('-embeddingSize', default=128, type=int)
        parser.add_argument('-hingeMargin', default=3, type=int)
        parser.add_argument('-mode', default='spatial_temporal',
                            help='four mode: cnn-rnn, spatial, temporal, spatial_temporal')
        # self.args = parser.parse_self.args()
        self.args = parser.parse_args()

        self.args.spatial = 0
        self.args.temporal = 0

        if self.args.mode == 'cnn-rnn':
            self.args.spatial = 0
            self.args.temporal = 0
        elif self.args.mode == 'spatial':
            self.args.spatial = 1
            self.args.temporal = 0
        elif self.args.mode == 'temporal':
            self.args.spatial = 0
            self.args.temporal = 1
        elif self.args.mode == 'spatial_temporal':
            self.args.spatial = 1
            self.args.temporal = 1
        else:
            print('Unknown mode')
            exit(0)

    def train(self):
        homeDir = os.getcwd()
        file_suffix = 'png'
        if self.args.dataset == 0:
            seq_root_rgb = 'data/i-ILDS-VID-DEBUG/sequences/'
            seq_root_of = 'data/i-ILDS-VID-DEBUG/hvp/sequences/'
        elif self.args.dataset == 1:
            seq_root_rgb = 'data/i-LIDS-VID/sequences/'
            seq_root_of = 'data/i-LIDS-VID-OF-HVP/sequences/'
        elif self.args.dataset == 2:
            seq_root_rgb = 'data/PRID2011/multi_shot/'
            seq_root_of = 'data/PRID2011-OF-HVP/multi_shot/'
        elif self.args.dataset == 3:
            seq_root_rgb = 'data/MARS/sequences/'
            seq_root_of = 'data/MARS-OF-HVP/sequences/'
            file_suffix = 'jpg'
        else:
            print('Unknown datasets')
            exit(0)

        print('loading Dataset - ', seq_root_rgb, seq_root_of)
        prepare_data = datasets.Prepare(self.args)
        dataset = prepare_data.prepareDataset(seq_root_rgb, seq_root_of, file_suffix)
        print('randomizing test/training split')
        utils = datasets.Utils()
        train_idx, test_idx = utils.partitionDataset(len(dataset), 0.5)

        # build the model
        n_person_train = len(train_idx)
        feature_net = Net(self.args, 16, self.args.nConvFilters, self.args.nConvFilters, n_person_train)
        train_seq = train.TrainSequence(self.args, train_idx, test_idx)
        train_seq.train_sequence(feature_net, dataset)


if __name__ == '__main__':
    tr = ReidTrain()
    tr.train()
