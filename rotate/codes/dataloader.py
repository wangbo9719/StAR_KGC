#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset
from os.path import join
import json
import random
import collections

class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode, entity2id=None, relation2id=None, data_dir=None, typecons = False):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.data_dir = data_dir
        self.typecons = typecons
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        # self.type_dict = self.get_type(self.data_dir)
        
    def __len__(self):
        return self.len

    def get_type(self, data_dir):

        with open(join(self.data_dir, "typecons.json"), "r") as f:
            type_dict = json.load(f)

        type_dict_id = collections.defaultdict(dict)
        for key in type_dict.keys():
            head_txt_list = list(type_dict[key]["head"])
            tail_txt_list = list(type_dict[key]["tail"])
            type_dict_id[key]["head"] = set([self.entity2id[ent] for ent in head_txt_list])
            type_dict_id[key]["tail"] = set([self.entity2id[ent] for ent in tail_txt_list])
        return type_dict_id


    def typecons_neg_sampling(self, triple):
        head, relation, tail = triple
        negative_sample_list = []
        negative_sample_size = 0


        while negative_sample_size < self.negative_sample_size:
            prob = random.random()
            negative_sample = None
            if self.mode == 'head-batch':
                if prob < 0.5:  # sample in domain
                    max_itex = 1000
                    while max_itex != 0:
                        negative_sample = random.choice(self.type_dict[relation]["head"])
                        max_itex -= 1
                        if negative_sample not in self.true_head[(relation, tail)]:
                            break
                else:
                    max_itex = 1000
                    while max_itex != 0:
                        negative_sample = random.randint(self.nentity)
                        max_itex -= 1
                        if negative_sample not in self.true_head[(relation, tail)] and negative_sample not in self.type_dict[relation]["head"]:
                            break
            elif self.mode == "tail-batch":
                if prob < 0.5:  # sample in domain
                    max_itex = 1000
                    while max_itex != 0:
                        negative_sample = random.choice(self.type_dict[relation]["tail"])
                        max_itex -= 1
                        if negative_sample not in self.true_tail[(head, relation)]:
                            break
                else:
                    max_itex = 1000
                    while max_itex != 0:
                        negative_sample = random.randint(self.nentity)
                        max_itex -= 1
                        if negative_sample not in self.true_tail[(head, relation)] and negative_sample not in self.type_dict[relation]["head"]:
                            break
            assert negative_sample != None
            negative_sample_list.append(negative_sample)
            negative_sample_size += 1
        return negative_sample_list



    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        if self.typecons == True:
            negative_sample_list = self.typecons_neg_sampling(positive_sample)
        else:
            negative_sample_list = []
            negative_sample_size = 0

            while negative_sample_size < self.negative_sample_size:
                negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
                if self.mode == 'head-batch':
                    mask = np.in1d(
                        negative_sample,
                        self.true_head[(relation, tail)],
                        assume_unique=True,
                        invert=True
                    )
                elif self.mode == 'tail-batch':
                    mask = np.in1d(
                        negative_sample,
                        self.true_tail[(head, relation)],
                        assume_unique=True,
                        invert=True
                    )
                else:
                    raise ValueError('Training batch mode %s not supported' % self.mode)
                negative_sample = negative_sample[mask]
                negative_sample_list.append(negative_sample)
                negative_sample_size += negative_sample.size

            negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        negative_sample = torch.from_numpy(negative_sample)
        
        positive_sample = torch.LongTensor(positive_sample)
            
        return positive_sample, negative_sample, subsampling_weight, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode
    
    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail

    
class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            # because tmp[tail] is (1, tail) in last sequence (it is in all_true_triples)
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)
            
        tmp = torch.LongTensor(tmp)            
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))
            
        return positive_sample, negative_sample, filter_bias, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode
    
class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data