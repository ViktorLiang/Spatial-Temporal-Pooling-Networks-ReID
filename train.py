import random as rand

import pickle

from models.spatial_temporal import *
import torch.optim as optim


class TrainSequence:
    def __init__(self, args, train_idx, test_idx):
        self.args = args
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.n_tran_psn = len(train_idx)

    def train_sequence(self, model, dataset):
        criterion = Criterion(self.args.hingeMargin)
        optimizer = optim.SGD(model.parameters(), lr=self.args.learningRate, momentum=self.args.momentum)
        model.cuda()

        for eph in range(self.args.nEpochs):
            total_loss = []
            for i in range(2 * self.n_tran_psn):
                # positive pair
                target = []
                if i % 2 == 0:
                    psn_id_a = psn_id_b = i / 2
                    target = [1, psn_id_a, psn_id_b]
                else:
                    psn_rand_id = torch.randperm(self.n_tran_psn)
                    target = [-1, psn_rand_id[0], psn_rand_id[1]]

                shape1 = dataset[psn_id_a][1].shape
                shape2 = dataset[psn_id_b][2].shape

                actual_size1 = self.args.sampleSeqLength if shape1[0] >= self.args.sampleSeqLength else shape1[0]
                actual_size2 = self.args.sampleSeqLength if shape2[0] >= self.args.sampleSeqLength else shape2[0]
                idx1 = rand.randint(0, shape1[0] - actual_size1 - 1) + 1
                idx2 = rand.randint(0, shape2[0] - actual_size2 - 1) + 1
                netinput_a = dataset[psn_id_a][1][idx1:idx1 + actual_size1]
                netinput_b = dataset[psn_id_b][2][idx2:idx2 + actual_size2]
                feature_p, feature_g, identity_p, identity_g = model(netinput_a, netinput_b)
                loss = criterion(feature_p, feature_g, identity_p, identity_g, target)
                total_loss.append(loss)
                optimizer.zero_grad()
                loss.backward()
                # hing_loss.backwrad()
                # lsoft_loss_p.backward()
                # lsoft_loss_g.backward()
                optimizer.step()
                # if eph % self.args.samplingEpochs == 0:
                # do test
                # do test
                # pass
            total_loss.append(loss)
            if eph % 20 == 0:
                torch.save(model.state_dict(), './model_saved/model-epoch-%s.pth' % eph)
            print('epoch %d ---- loss %f' % (eph, loss))
        pickle.dump(total_loss, open('./loss.log', 'wb'))

    def test(self):
        pass
