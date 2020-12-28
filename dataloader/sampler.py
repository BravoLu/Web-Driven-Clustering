import collections
import os
import random
from torch.utils.data import dataset, sampler
from torchvision.datasets.folder import default_loader
import numpy as np
from collections import defaultdict
import copy


class RandomIdSampler(sampler.Sampler):
    @staticmethod
    def _sample(population, k):
        if len(population) < k:
            population = population * k
        return random.sample(population, k)

    def __init__(self, data_source, batch_image):
        super(RandomIdSampler, self).__init__(data_source)
        self.data_source = data_source
        self.batch_image = batch_image
        
        path_list = []
        ID_list = []
        self._id2index = collections.defaultdict(list)
        for idx, trainset in enumerate(data_source):
            path_list.append(trainset[0])
            ID_list.append(trainset[1])
        
        self.unique_ids = sorted(set(ID_list))
        
        #print("unique_ids:%s"%(len(self.unique_ids)))
        for idx, path in enumerate(path_list):
            _id = ID_list[idx]
            self._id2index[_id].append(idx)   

    def __iter__(self):
        unique_ids = self.unique_ids
        random.shuffle(unique_ids)

        imgs = []
        for _id in unique_ids:
            imgs.extend(self._sample(self._id2index[_id], self.batch_image))
        return iter(imgs)

    def __len__(self):
        return len(self._id2index) * self.batch_image



class RandomIdentitySampler(sampler.Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances, name=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        #changed according to the dataset
        for index, inputs in enumerate(self.data_source):
            self.index_dic[inputs[1]].append(index)

        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length

