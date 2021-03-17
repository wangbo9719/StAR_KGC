import torch
import numpy as np
from torch.utils.data import DataLoader
from dataloader import TestDataset
import logging
import os
import collections
from tqdm import tqdm

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def complete_embs(model, entity2id, relation2id, nentity, args):
    train_dataset = read_triple(os.path.join(args.data_path, 'train_1900.txt'), entity2id, relation2id)
    test_dataset = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    removed_dataset = read_triple(os.path.join(args.data_path, 'train_removed.txt'), entity2id, relation2id)

    train_ents = set()
    for triple in train_dataset:
        h, _, t = triple
        train_ents.add(h)
        train_ents.add(t)
    test_ents = set()
    for triple in test_dataset:
        h, _, t = triple
        test_ents.add(h)
        test_ents.add(t)
    removed_ents = set()
    for triple in removed_dataset:
        h, _, t = triple
        removed_ents.add(h)
        removed_ents.add(t)

    unseen_ents = list(test_ents - train_ents)

    for ent in unseen_ents:
        ent_emb = model.entity_embedding.data[ent]
        if ent in removed_ents:
            h_tmp_triples = []
            t_tmp_triples = []
            for triple in removed_dataset:
                _h, _r, _t = triple
                if _h == ent:
                    h_tmp_triples.append(triple)
                elif _t == ent:
                    t_tmp_triples.append(triple)
            if len(h_tmp_triples) != 0:
                for _tri in h_tmp_triples:
                    _h, _r, _t = _tri
                    ent_emb += model.entity_embedding.data[_t] - model.entity_embedding.data[_r]
                    # ent_emb += torch.index_select(model.entity_embedding.data, dim=0,index=_t) - torch.index_select(model.relation_embedding.data, dim=0,index=_r)
            if len(t_tmp_triples) != 0:
                for _tri in t_tmp_triples:
                    _h, _r, _t = _tri
                    ent_emb += model.entity_embedding.data[_h] + model.entity_embedding.data[_r]
                    # ent_emb += torch.index_select(model.entity_embedding.data, dim=0,index=_h) + torch.index_select(model.relation_embedding.data, dim=0,index=_r)
            ent_emb = ent_emb / (len(h_tmp_triples) + len(t_tmp_triples))
            model.entity_embedding.data[ent] = ent_emb

