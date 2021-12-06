import torch.utils.data
import io, os
import pandas as pd
import time
import numpy as np
import sys
import config
import math
import itertools

opt = config.OpportunityOpt
WIN_LEN = opt['seq_len']

class OpportunityDataset(torch.utils.data.Dataset):
    def __init__(self, file, users=None, positions=None, activities=None):
        st = time.time()

        self.users = users
        self.positions = positions
        self.df = pd.read_csv(file)

        if users is not None:
            cond_list = []
            for d in users:
                cond_list.append('User == "{:s}"'.format(d))
            cond_str = ' | '.join(cond_list)
            self.df = self.df.query(cond_str)
        if positions is not None:
            cond_list = []
            for d in positions:
                cond_list.append('Position == "{:s}"'.format(d))
            cond_str = ' | '.join(cond_list)
            self.df = self.df.query(cond_str)
        if activities is not None:
            cond_list = []
            for d in activities:
                cond_list.append('Activity == "{:s}"'.format(d))
            cond_str = ' | '.join(cond_list)
            self.df = self.df.query(cond_str)

        self.domains = sorted(list(itertools.product(users, positions)))
        ppt = time.time()

        self.preprocessing()
        print('Loading data done with rows:{:d}\tPreprocessing:{:f}\tTotal Time:{:f}'.format(len(self.df.index),
                                                                                             time.time() - ppt,
                                                                                             time.time() - st))

    def preprocessing(self):
        self.features = []
        self.class_labels = []
        # self.domain_labels = []
        self.position_labels = []

        self.datasets = []  # list of dataset per each domain

        for idx in range(max(len(self.df) // WIN_LEN, 0)):
            user = self.df.iloc[idx * WIN_LEN, 6]
            position_label = self.df.iloc[idx * WIN_LEN, 7]
            class_label = self.df.iloc[idx * WIN_LEN, 8]
            # domain_label = self.domains.index((user, position))

            feature = self.df.iloc[idx * WIN_LEN:(idx + 1) * WIN_LEN, 0:6].values
            feature = feature.T

            self.features.append(feature)
            self.position_labels.append(self.position_to_number(position_label))
            self.class_labels.append(self.class_to_number(class_label))
            # self.domain_labels.append(domain_label)

        self.features = np.array(self.features, dtype=np.float)
        self.class_labels = np.array(self.class_labels)
        self.position_labels = np.array(self.position_labels)
        # self.domain_labels = np.array(self.domain_labels)


        self.datasets = torch.utils.data.TensorDataset(torch.from_numpy(self.features).float(),
                                                       torch.from_numpy(self.class_labels),
                                                       torch.from_numpy(self.position_labels))
        # type(self.datasets) => torch.utils.data.datasets.TensorDataset

        """
        # previous code on domains: but we don't need them now
        # append dataset for each domain
        for domain_idx in range(len(self.domains)):
            indices = np.where(self.domain_labels == domain_idx)[0]
            self.datasets.append(torch.utils.data.TensorDataset(torch.from_numpy(self.features[indices]).float(),
                                                                torch.from_numpy(self.class_labels[indices]),
                                                                # torch.from_numpy(self.domain_labels[indices]))
                                                                torch.from_numpy(self.position_labels[indices]))
        """
        # concated dataset
        # self.dataset = torch.utils.data.ConcatDataset(self.datasets)
        # self.dataset = self.datasets ??


    def class_to_number(self, label):
        dic = {
            'stand': 0,
            'walk': 1,
            'sit': 2,
            'lie': 3
        }
        return dic[label]

    def position_to_number(self, position):
        # 'RUA', 'LLA', 'L_Shoe', 'Back' ==> selected attribute positions
        dic = {
            'RUA': 0,
            'LLA': 1,
            'L_Shoe': 2,
            'Back': 3
        }
        return dic[position]


    def __len__(self):
        return len(self.datasets)

    def get_num_domains(self):
        return len(self.domains)

    def __getitem__(self, idx):
        # if isinstance(idx, torch.Tensor):
        #     idx = idx.item()

        # feature, cl, pl = self.dataset[idx] # class_label, position_label
        feature, cl, pl = self.datasets[idx]
        feature = feature.float()
        feature = torch.transpose(feature, 0, 1)
        feature = feature.unsqueeze(0)

        one_hot_position_label = torch.zeros(4)
        one_hot_position_label[(pl-1)] = torch.tensor(1)

        # one_hot_cl = torch.zeros(opt['num_class']) # number of classes
        # one_hot_cl[(cl-1)] = torch.tensor(1)

        # print("feature size: ", feature.size(), "cl size: ", cl.size(), "cl: ", cl, "pl size: ", pl.size())

        # might need to change the feature dimension
        # return feature, one_hot_cl
        return feature, one_hot_position_label, cl # order (feature, one_hot_position_label, class_label)


def get_loader(sensor_data_file_path, users, positions, activities, batch_size, dataset, mode, num_workers=1):
    """ Build and return a data loader """
    if dataset == 'Opportunity':
        dataset = OpportunityDataset(file=sensor_data_file_path, users=users, positions=positions, activities = activities)

    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=(mode == 'train'), num_workers=num_workers)
    print("loader done!")
    return data_loader


if __name__ == '__main__':
    # src_users = list(np.random.permutation(config.OpportunityOpt['users'])[:2])
    # src_positions = list(np.random.permutation(config.OpportunityOpt['positions'])[:5])
    #
    # tgt_users = list(set(config.OpportunityOpt['users']) - set(src_users))
    # tgt_positions = list(set(config.OpportunityOpt['positions']) - set(src_positions))
    #
    # src_domains = sorted(list(itertools.product(src_users, src_positions)))
    # tgt_domains = sorted(list(itertools.product(tgt_users, tgt_positions)))
    #
    # print("Src users: ", src_users, ", src positions: ", src_positions)


    # read for source
    oppDataset = OpportunityDataset(file='/mnt/sting/adiorz/mobile_sensing/datasets/opportunity_std_scaled_all.csv',
                                    users = config.OpportunityOpt['users'], positions=['RUA', 'LLA', 'L_Shoe', 'Back'])
    print("Done!")


