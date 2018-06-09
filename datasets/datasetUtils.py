import torch


class Utils:
    def partitionDataset(self, n_persons, train_split):
        split_point = int(n_persons * train_split)
        inds = torch.randperm(n_persons)
        train_idx = inds[:split_point]
        test_idx = inds[split_point:]
        return train_idx, test_idx
