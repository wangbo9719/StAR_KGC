from peach.help import *
import argparse
import collections
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from os.path import join
from peach.common import load_list_from_file, load_tsv, file_exists, dir_exists, save_json, load_json
from tqdm import tqdm, trange
from transformers import BertConfig, BertTokenizer, RobertaConfig, RobertaTokenizer
from kbc.kb_dataset import KbDataset
from kbc.models import BertForPairScoring, RobertaForPairScoring
from kbc.metric import calculate_metrics_for_link_prediction, safe_ranking
from kbc.utils_fn import train
import numpy as np


def data_collate_fn_general(batch, pad_id=0):
    tensors_list = list(zip(*batch))
    return_list = []
    for _idx_t, _tensors in enumerate(tensors_list):
        if _idx_t == 0:
            padding_value = pad_id
        else:
            padding_value = 0

        if isinstance(_tensors[0], str):
            return_list.append(list(_tensors))
        elif _tensors[0].dim() >= 1:
            return_list.append(
                torch.nn.utils.rnn.pad_sequence(_tensors, batch_first=True, padding_value=padding_value),
            )
        else:
            return_list.append(torch.stack(_tensors, dim=0))
    return tuple(return_list)

class DatasetForPairwiseRankingLP(KbDataset):
    def __init__(self, *arg, **kwargs):
        super(DatasetForPairwiseRankingLP, self).__init__(*arg, **kwargs)

    def assemble_conponents(self, head_ids, rel_ids, tail_ids):
        max_ent_len = self.max_seq_length - 3 - len(rel_ids)
        head_ids = head_ids[:max_ent_len]
        tail_ids = tail_ids[:max_ent_len]

        src_input_ids = [self._cls_id] + head_ids + [self._sep_id] + rel_ids + [self._sep_id]
        src_mask_ids = [1] * len(src_input_ids)
        src_segment_ids = [0] * (len(head_ids) + 2) + [1] * (len(rel_ids) + 1)

        tgt_input_ids = [self._cls_id]  + tail_ids + [self._sep_id]
        tgt_mask_ids = [1] * len(tgt_input_ids)
        tgt_segment_ids = [0] * (len(tail_ids) + 2)

        assert len(tgt_segment_ids) <= 512

        return (src_input_ids, src_mask_ids, src_segment_ids), (tgt_input_ids, tgt_mask_ids, tgt_segment_ids)

    def __getitem__(self, item):
        if self.data_type == "train":  # this is for negative sampling
            assert self.subj2objs is not None and self.obj2subjs is not None
        pos_raw_ex = self.raw_examples[item]
        # negative sampling
        neg_raw_ex_set = set()
        neg_raw_ex_list = []
        neg_raw_ex_str_set = set()
        tolerate = 200
        while len(neg_raw_ex_str_set) < self.neg_times and tolerate > 0:
            neg_raw_ex = self.negative_sampling(pos_raw_ex, self.neg_weights)
            neg_raw_ex_str = str(neg_raw_ex)
            if neg_raw_ex_str not in neg_raw_ex_str_set:
                neg_raw_ex_list.append(neg_raw_ex)
                neg_raw_ex_str_set.add(neg_raw_ex_str)
            tolerate -= 1
        # assert len(neg_raw_ex_list) == 0

        if len(neg_raw_ex_list) < self.neg_times:
            neg_raw_ex_list = [neg_raw_ex_list[idx%len(neg_raw_ex_list)] for idx in range(self.neg_times)]

        # ids
        (src_input_ids, src_mask_ids, src_segment_ids), \
        (tgt_input_ids, tgt_mask_ids, tgt_segment_ids) \
            = self.assemble_conponents(*self.convert_raw_example_to_features(pos_raw_ex, method="5"))


        neg_data_list = []
        for neg_raw_ex in neg_raw_ex_list:
            neg_data_p1, neg_data_p2 = self.assemble_conponents(
                *self.convert_raw_example_to_features(neg_raw_ex, method="5"))
            neg_data = list(neg_data_p1) + list(neg_data_p2)
            neg_data = [torch.tensor(_ids, dtype=torch.long) for _ids in neg_data]
            neg_data_list.append(neg_data)

        virtual_batch = list(zip(*neg_data_list))
        # neg_times, sl
        neg_src_input_ids, neg_src_mask_ids, neg_src_segment_ids, \
        neg_tgt_input_ids, neg_tgt_mask_ids, neg_tgt_segment_ids = virtual_batch


        return (
            torch.tensor(src_input_ids, dtype=torch.long),
            torch.tensor(src_mask_ids, dtype=torch.long),
            torch.tensor(src_segment_ids, dtype=torch.long),
            torch.tensor(tgt_input_ids, dtype=torch.long),
            torch.tensor(tgt_mask_ids, dtype=torch.long),
            torch.tensor(tgt_segment_ids, dtype=torch.long),
            neg_src_input_ids,
            neg_src_mask_ids,
            neg_src_segment_ids,
            neg_tgt_input_ids,
            neg_tgt_mask_ids,
            neg_tgt_segment_ids
        )

    def data_collate_fn(self, batch):
        tensors_list = list(zip(*batch)) # 12 * bs * 1/neg_times * sl
        return_list = []
        for _idx_t, _tensors in enumerate(tensors_list):
            if _idx_t % 3 == 0:
                padding_value = self._pad_id
            else:
                padding_value = 0

            if _idx_t >= 6:  # _tensors : bs * neg_times * sl
                # 2D padding
                _max_len_last_dim = 0
                # _tensors : bs * neg_times * sl
                # _tensor :  tuple, neg_times * sl
                for _tensor in _tensors:
                    _local_max_len_last_dim = max(len(_t) for _t in list(_tensor))
                    _max_len_last_dim = max(_max_len_last_dim, _local_max_len_last_dim)
                # padding
                _new_tensors = []
                for _tensor in _tensors:
                    inner_tensors = []
                    for idx, _ in enumerate(list(_tensor)):
                        _pad_shape = _max_len_last_dim - len(_tensor[idx])
                        _pad_tensor = torch.tensor([padding_value] * _pad_shape, device=_tensor[idx].device, dtype=_tensor[idx].dtype)
                        _new_inner_tensor = torch.cat([_tensor[idx], _pad_tensor], dim=0)
                        inner_tensors.append(_new_inner_tensor)
                    _tensors_tuple = tuple(ts for ts in inner_tensors)
                    _new_tensors.append(torch.stack(_tensors_tuple, dim=0))
                return_list.append(
                    torch.nn.utils.rnn.pad_sequence(_new_tensors, batch_first=True, padding_value=padding_value),
                )
            else:
                if _tensors[0].dim() >= 1:
                    return_list.append(
                        torch.nn.utils.rnn.pad_sequence(_tensors, batch_first=True, padding_value=padding_value),
                    )
                else:
                    return_list.append(torch.stack(_tensors, dim=0))
        return tuple(return_list)


    def __len__(self):
        return len(self.raw_examples)

    @classmethod
    def batch2feed_dict(cls, batch, data_format=None):
        inputs = {
            'src_input_ids': batch[0],  # bs, sl
            'src_attention_mask': batch[1],  #
            'src_token_type_ids': batch[2],  #
            'tgt_input_ids': batch[3],  # bs, sl
            'tgt_attention_mask': batch[4],  #
            'tgt_token_type_ids': batch[5],  #
            "label_dict": {
                'neg_src_input_ids': batch[6],  # bs, sl
                'neg_src_attention_mask': batch[7],  #
                'neg_src_token_type_ids': batch[8],  #
                'neg_tgt_input_ids': batch[9],  # bs, sl
                'neg_tgt_attention_mask': batch[10],  #
                'neg_tgt_token_type_ids': batch[11],  #
            }
        }
        return inputs

def predict(args, raw_examples, dataset_list, model, verbose=True):
    logging.info("***** Running Prediction*****")
    model.eval()
    # get the last one (i.e., test) to make use if its useful functions and data
    standard_dataset = dataset_list[-1]

    ents = set()
    g_subj2objs = collections.defaultdict(lambda: collections.defaultdict(set))
    g_obj2subjs = collections.defaultdict(lambda: collections.defaultdict(set))
    for _ds in dataset_list:
        for _raw_ex in _ds.raw_examples:
            _head, _rel, _tail = _raw_ex
            ents.add(_head)
            ents.add(_tail)
            g_subj2objs[_head][_rel].add(_tail)
            g_obj2subjs[_tail][_rel].add(_head)
    ent_list = list(sorted(ents))
    rel_list = list(sorted(standard_dataset.rel_list))

    # ========= get all embeddings ==========
    print("get all embeddings")

    save_dir = args.model_name_or_path if dir_exists(args.model_name_or_path) else args.output_dir
    save_path = os.path.join(save_dir, "saved_emb_mat.np")

    if dir_exists(save_dir) and file_exists(save_path):
        print("\tload from file")
        emb_mat = torch.load(save_path)
    else:
        print("\tget all ids")
        input_ids_list, mask_ids_list, segment_ids_list = [], [], []
        for _ent in tqdm(ent_list):
            for _idx_r, _rel in enumerate(rel_list):
                head_ids, rel_ids, tail_ids = standard_dataset.convert_raw_example_to_features(
                    [_ent, _rel, _ent], method="4")
                head_ids, rel_ids, tail_ids = head_ids[1:-1], rel_ids[1:-1], tail_ids[1:-1]
                # truncate
                max_ent_len = standard_dataset.max_seq_length - 3 - len(rel_ids)
                head_ids = head_ids[:max_ent_len]
                tail_ids = tail_ids[:max_ent_len]

                src_input_ids = [standard_dataset._cls_id] + head_ids + [standard_dataset._sep_id] + rel_ids + [
                    standard_dataset._sep_id]
                src_mask_ids = [1] * len(src_input_ids)
                src_segment_ids = [0] * (len(head_ids) + 2) + [1] * (len(rel_ids) + 1)

                if _idx_r == 0:
                    tgt_input_ids = [standard_dataset._cls_id] + tail_ids + [standard_dataset._sep_id]
                    tgt_mask_ids = [1] * len(tgt_input_ids)
                    tgt_segment_ids = [0] * (len(tail_ids) + 2)
                    input_ids_list.append(tgt_input_ids)
                    mask_ids_list.append(tgt_mask_ids)
                    segment_ids_list.append(tgt_segment_ids)

                input_ids_list.append(src_input_ids)
                mask_ids_list.append(src_mask_ids)
                segment_ids_list.append(src_segment_ids)

        # # padding
        max_len = max(len(_e) for _e in input_ids_list)
        assert max_len <= standard_dataset.max_seq_length
        input_ids_list = [_e + [standard_dataset._pad_id] * (max_len - len(_e)) for _e in input_ids_list]
        mask_ids_list = [_e + [0] * (max_len - len(_e)) for _e in mask_ids_list]
        segment_ids_list = [_e + [0] * (max_len - len(_e)) for _e in segment_ids_list]
        # # dataset
        enc_dataset = TensorDataset(
            torch.tensor(input_ids_list, dtype=torch.long),
            torch.tensor(mask_ids_list, dtype=torch.long),
            torch.tensor(segment_ids_list, dtype=torch.long),
        )
        enc_dataloader = DataLoader(
            enc_dataset, sampler=SequentialSampler(enc_dataset) , batch_size=args.eval_batch_size*2)
        print("\tget all emb via model")
        embs_list = []
        for batch in tqdm(enc_dataloader, desc="entity embedding", disable=(not verbose)):
            batch = tuple(t.to(args.device) for t in batch)
            _input_ids, _mask_ids, _segment_ids = batch
            with torch.no_grad():
                embs = model.encoder(_input_ids, attention_mask=_mask_ids, token_type_ids=_segment_ids)
                embs = embs.detach().cpu()
                embs_list.append(embs)

        emb_mat = torch.cat(embs_list, dim=0).contiguous()
        assert emb_mat.shape[0] == len(input_ids_list)
        # save emb_mat
        if dir_exists(save_dir):
            torch.save(emb_mat, save_path)

    # # assign to ent
    assert len(ent_list) *(1+len(rel_list)) == emb_mat.shape[0]

    ent_rel2emb = collections.defaultdict(dict)
    ent2emb = dict()

    ptr_row = 0
    for _ent in ent_list:
        for _idx_r, _rel in enumerate(rel_list):
            if _idx_r == 0:
                ent2emb[_ent] = emb_mat[ptr_row]
                ptr_row += 1
            ent_rel2emb[_ent][_rel] = emb_mat[ptr_row]
            ptr_row += 1
    # ========= run link prediction ==========

    # * begin to get hit
    ranks_left, ranks_right, ranks = [], [], []
    hits_left, hits_right, hits = [], [], []
    top_ten_hit_count = 0
    for i in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])

    for _idx_ex, _triplet in enumerate(tqdm(raw_examples, desc="evaluating")):
        _head, _rel, _tail = _triplet

        head_ent_list = []
        tail_ent_list = []

        # head corrupt
        _pos_head_ents = g_obj2subjs[_tail][_rel]
        _neg_head_ents = ents - _pos_head_ents
        # -------------------------------------------
        # _tail_s = set()
        # _tail_s.add(_tail)
        # _neg_head_ents = _neg_head_ents - _tail_s
        # -------------------------------------------
        head_ent_list.append(_head)  # positive example
        head_ent_list.extend(_neg_head_ents)  # negative examples
        tail_ent_list.extend([_tail] * (1 + len(_neg_head_ents)))
        split_idx = len(head_ent_list)

        # tail corrupt
        _pos_tail_ents = g_subj2objs[_head][_rel]
        _neg_tail_ents = ents - _pos_tail_ents
        # -------------------------------------------
        # _head_s = set()
        # _head_s.add(_head)
        # _neg_tail_ents = _neg_tail_ents - _head_s
        # -------------------------------------------
        head_ent_list.extend([_head] * (1 + len(_neg_tail_ents)))
        tail_ent_list.append(_tail)  # positive example
        tail_ent_list.extend(_neg_tail_ents)  # negative examples

        triplet_list = list([_h, _rel, _t] for _h, _t in zip(head_ent_list, tail_ent_list))

        # build dataset
        rep_src_list = [ent_rel2emb[_h][_rel] for _h, _rel, _ in triplet_list]
        rep_tgt_list = [ent2emb[_t] for _, _rel, _t in triplet_list]
        all_rep_src = torch.stack(rep_src_list, dim=0).to(args.device)
        all_rep_tgt = torch.stack(rep_tgt_list, dim=0).to(args.device)

        local_scores_list = []
        sim_batch_size = args.eval_batch_size * 8
        if args.cls_method == "dis":
            for _idx_r in range(0, all_rep_src.shape[0], sim_batch_size):
                _rep_src, _rep_tgt = all_rep_src[_idx_r: _idx_r + sim_batch_size], all_rep_tgt[
                                                                                   _idx_r: _idx_r + sim_batch_size]
                with torch.no_grad():
                    distances = model.distance_metric_fn(_rep_src, _rep_tgt)
                    local_scores = - distances
                    local_scores = local_scores.detach().cpu().numpy()
                local_scores_list.append(local_scores)
        elif args.cls_method == "cls":
            for _idx_r in range(0, all_rep_src.shape[0], sim_batch_size):
                _rep_src, _rep_tgt = all_rep_src[_idx_r: _idx_r + sim_batch_size], all_rep_tgt[
                                                                                   _idx_r: _idx_r + sim_batch_size]
                with torch.no_grad():
                    logits = model.classifier(_rep_src, _rep_tgt)
                    logits = torch.softmax(logits, dim=-1)
                    local_scores = logits.detach().cpu().numpy()[:, 1]
                local_scores_list.append(local_scores)
        scores = np.concatenate(local_scores_list, axis=0)

        # left
        left_scores = scores[:split_idx]
        left_rank = safe_ranking(left_scores)
        ranks_left.append(left_rank)
        ranks.append(left_rank)

        # right
        right_scores = scores[split_idx:]
        right_rank = safe_ranking(right_scores)
        ranks_right.append(right_rank)
        ranks.append(right_rank)

        # log
        top_ten_hit_count += (int(left_rank <= 10) + int(right_rank <= 10))
        if (_idx_ex + 1) % 10 == 0:
            logger.info("hit@10 until now: {}".format(top_ten_hit_count * 1.0 / len(ranks)))
            logger.info('mean rank until now: {}'.format(np.mean(ranks)))

        # hits
        for hits_level in range(10):
            if left_rank <= hits_level + 1:

                hits[hits_level].append(1.0)
                hits_left[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)
                hits_left[hits_level].append(0.0)

            if right_rank <= hits_level + 1:
                hits[hits_level].append(1.0)
                hits_right[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)
                hits_right[hits_level].append(0.0)
    if verbose:
        for i in [0, 2, 9]:
            logger.info('Hits left @{0}: {1}'.format(i + 1, np.mean(hits_left[i])))
            logger.info('Hits right @{0}: {1}'.format(i + 1, np.mean(hits_right[i])))
            logger.info('Hits @{0}: {1}'.format(i + 1, np.mean(hits[i])))
        logger.info('Mean rank left: {0}'.format(np.mean(ranks_left)))
        logger.info('Mean rank right: {0}'.format(np.mean(ranks_right)))
        logger.info('Mean rank: {0}'.format(np.mean(ranks)))
        logger.info('Mean reciprocal rank left: {0}'.format(np.mean(1. / np.array(ranks_left))))
        logger.info('Mean reciprocal rank right: {0}'.format(np.mean(1. / np.array(ranks_right))))
        logger.info('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))

        tuple_ranks = [[int(_l), int(_r)] for _l, _r in zip(ranks_left, ranks_right)]
        return tuple_ranks

def predict_NELL(args, raw_examples, dataset_list, model, verbose=True):
    logging.info("***** Running Prediction*****")
    model.eval()
    # get the last one (i.e., test) to make use if its useful functions and data
    standard_dataset = dataset_list[-1]

    ents = set()
    g_subj2objs = collections.defaultdict(lambda: collections.defaultdict(set))
    g_obj2subjs = collections.defaultdict(lambda: collections.defaultdict(set))
    for _ds in dataset_list:
        for _raw_ex in _ds.raw_examples:
            _head, _rel, _tail = _raw_ex
            ents.add(_head)
            ents.add(_tail)
            g_subj2objs[_head][_rel].add(_tail)
            g_obj2subjs[_tail][_rel].add(_head)
    ent_list = list(sorted(ents))
    # rel_list = list(sorted(standard_dataset.rel_list))
    rel_set = set()
    for triplet in raw_examples:
        _, r, _ = triplet
        rel_set.add(r)
    rel_list = list(sorted(list(rel_set)))

    # ========= get all embeddings ==========
    print("get all embeddings")

    save_dir = args.model_name_or_path if dir_exists(args.model_name_or_path) else args.output_dir
    save_path = os.path.join(save_dir, "saved_emb_mat.np")

    if dir_exists(save_dir) and file_exists(save_path):
        print("\tload from file")
        emb_mat = torch.load(save_path)
    else:
        print("\tget all ids")
        input_ids_list, mask_ids_list, segment_ids_list = [], [], []
        for _ent in tqdm(ent_list):
            for _idx_r, _rel in enumerate(rel_list):
                head_ids, rel_ids, tail_ids = standard_dataset.convert_raw_example_to_features(
                    [_ent, _rel, _ent], method="4")
                head_ids, rel_ids, tail_ids = head_ids[1:-1], rel_ids[1:-1], tail_ids[1:-1]
                # truncate
                max_ent_len = standard_dataset.max_seq_length - 3 - len(rel_ids)
                head_ids = head_ids[:max_ent_len]
                tail_ids = tail_ids[:max_ent_len]

                src_input_ids = [standard_dataset._cls_id] + head_ids + [standard_dataset._sep_id] + rel_ids + [
                    standard_dataset._sep_id]
                src_mask_ids = [1] * len(src_input_ids)
                src_segment_ids = [0] * (len(head_ids) + 2) + [1] * (len(rel_ids) + 1)

                if _idx_r == 0:
                    tgt_input_ids = [standard_dataset._cls_id] + tail_ids + [standard_dataset._sep_id]
                    tgt_mask_ids = [1] * len(tgt_input_ids)
                    tgt_segment_ids = [0] * (len(tail_ids) + 2)
                    input_ids_list.append(tgt_input_ids)
                    mask_ids_list.append(tgt_mask_ids)
                    segment_ids_list.append(tgt_segment_ids)

                input_ids_list.append(src_input_ids)
                mask_ids_list.append(src_mask_ids)
                segment_ids_list.append(src_segment_ids)

        # # padding
        max_len = max(len(_e) for _e in input_ids_list)
        assert max_len <= standard_dataset.max_seq_length
        input_ids_list = [_e + [standard_dataset._pad_id] * (max_len - len(_e)) for _e in input_ids_list]
        mask_ids_list = [_e + [0] * (max_len - len(_e)) for _e in mask_ids_list]
        segment_ids_list = [_e + [0] * (max_len - len(_e)) for _e in segment_ids_list]
        # # dataset
        enc_dataset = TensorDataset(
            torch.tensor(input_ids_list, dtype=torch.long),
            torch.tensor(mask_ids_list, dtype=torch.long),
            torch.tensor(segment_ids_list, dtype=torch.long),
        )
        enc_dataloader = DataLoader(
            enc_dataset, sampler=SequentialSampler(enc_dataset) , batch_size=args.eval_batch_size*2)
        print("\tget all emb via model")
        embs_list = []
        for batch in tqdm(enc_dataloader, desc="entity embedding", disable=(not verbose)):
            batch = tuple(t.to(args.device) for t in batch)
            _input_ids, _mask_ids, _segment_ids = batch
            with torch.no_grad():
                embs = model.encoder(_input_ids, attention_mask=_mask_ids, token_type_ids=_segment_ids)
                embs = embs.detach().cpu()
                embs_list.append(embs)

        emb_mat = torch.cat(embs_list, dim=0).contiguous()
        assert emb_mat.shape[0] == len(input_ids_list)
        # save emb_mat
        if dir_exists(save_dir):
            torch.save(emb_mat, save_path)

    # # assign to ent
    assert len(ent_list) *(1+len(rel_list)) == emb_mat.shape[0]

    ent_rel2emb = collections.defaultdict(dict)
    ent2emb = dict()


    ptr_row = 0
    for _ent in ent_list:
        for _idx_r, _rel in enumerate(rel_list):
            if _idx_r == 0:
                ent2emb[_ent] = emb_mat[ptr_row]
                ptr_row += 1
            ent_rel2emb[_ent][_rel] = emb_mat[ptr_row]
            ptr_row += 1

    # ========= run link prediction ==========

    # * begin to get hit
    ranks_left, ranks_right, ranks = [], [], []
    hits_left, hits_right, hits = [], [], []
    top_ten_hit_count = 0
    top_five_hit_count = 0
    top_one_hit_count = 0
    for i in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])
    for _idx_ex, _triplet in enumerate(tqdm(raw_examples, desc="evaluating")):
        _head, _rel, _tail = _triplet

        head_ent_list = []
        tail_ent_list = []

        # tail corrupt
        _pos_tail_ents = g_subj2objs[_head][_rel]
        _neg_tail_ents = ents - _pos_tail_ents
        _neg_tail_ents = [_ent for _ent in _neg_tail_ents if _ent in standard_dataset.type_dict[_rel]["tail"]]
        head_ent_list.extend([_head] * (1 + len(_neg_tail_ents)))
        tail_ent_list.append(_tail)  # positive example
        tail_ent_list.extend(_neg_tail_ents)  # negative examples

        triplet_list = list([_h, _rel, _t] for _h, _t in zip(head_ent_list, tail_ent_list))

        # build dataset
        rep_src_list = [ent_rel2emb[_h][_rel] for _h, _rel, _ in triplet_list]
        rep_tgt_list = [ent2emb[_t] for _, _rel, _t in triplet_list]
        all_rep_src = torch.stack(rep_src_list, dim=0).to(args.device)
        all_rep_tgt = torch.stack(rep_tgt_list, dim=0).to(args.device)

        local_scores_list = []
        sim_batch_size = args.eval_batch_size * 8
        if args.cls_method == "dis":
            for _idx_r in range(0, all_rep_src.shape[0], sim_batch_size):
                _rep_src, _rep_tgt = all_rep_src[_idx_r: _idx_r + sim_batch_size], all_rep_tgt[
                                                                                   _idx_r: _idx_r + sim_batch_size]
                with torch.no_grad():
                    distances = model.distance_metric_fn(_rep_src, _rep_tgt).to(torch.float32)
                    local_scores = - distances
                    local_scores = local_scores.detach().cpu().numpy()
                local_scores_list.append(local_scores)
        elif args.cls_method == "cls":
            for _idx_r in range(0, all_rep_src.shape[0], sim_batch_size):
                _rep_src, _rep_tgt = all_rep_src[_idx_r: _idx_r + sim_batch_size], all_rep_tgt[
                                                                                   _idx_r: _idx_r + sim_batch_size]
                with torch.no_grad():
                    logits = model.classifier(_rep_src, _rep_tgt).to(torch.float32)
                    logits = torch.softmax(logits, dim=-1)
                    local_scores = logits.detach().cpu().numpy()[:, 1]
                local_scores_list.append(local_scores)
        scores = np.concatenate(local_scores_list, axis=0)

        # right
        right_scores = scores
        right_rank = safe_ranking(right_scores)
        ranks_right.append(right_rank)
        ranks.append(right_rank)
        # log
        top_ten_hit_count += int(right_rank <= 10)
        top_five_hit_count += int(right_rank <= 5)
        top_one_hit_count += int(right_rank <= 1)
        if (_idx_ex + 1) % 10 == 0:
            logger.info("hit@1 until now: {}".format(top_one_hit_count * 1.0 / len(ranks)))
            logger.info("hit@5 until now: {}".format(top_five_hit_count * 1.0 / len(ranks)))
            logger.info("hit@10 until now: {}".format(top_ten_hit_count * 1.0 / len(ranks)))


        # hits
        for hits_level in range(10):

            if right_rank <= hits_level + 1:
                hits[hits_level].append(1.0)
                hits_right[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)
                hits_right[hits_level].append(0.0)


    if verbose:
        for i in [0, 4, 9]:
            logger.info('Hits right @{0}: {1}'.format(i + 1, np.mean(hits_right[i])))
        logger.info('Mean rank right: {0}'.format(np.mean(ranks_right)))
        logger.info('Mean reciprocal rank right: {0}'.format(np.mean(1. / np.array(ranks_right))))

        with open(join(args.output_dir, "link_prediction_metrics.txt"), "w", encoding="utf-8") as fp:
            for i in [0, 4, 9]:
                fp.write('Hits right @{0}: {1}\n'.format(i + 1, np.mean(hits_right[i])))
            fp.write('Mean rank right: {0}\n'.format(np.mean(ranks_right)))
            fp.write('Mean reciprocal rank right: {0}\n'.format(np.mean(1. / np.array(ranks_right))))
        print("save finished!")

        tuple_ranks = [[int(_l), int(_r)] for _l, _r in zip(ranks_left, ranks_right)]
        return tuple_ranks

def evaluate_pairwise_ranking(args, eval_dataset, model, tokenizer, global_step=None, file_prefix=""):
    def str2ids(text, max_len=None):
        if args.do_lower_case:
            text = text.lower()
        wps = tokenizer.tokenize(text)
        if max_len is not None:
            wps = tokenizer.tokenize(text)[:max_len]
        return tokenizer.convert_tokens_to_ids(wps)

    logging.info("***** Running evaluation at {}*****".format(global_step))
    logging.info("  Num examples = %d", len(eval_dataset))
    model.eval()

    # get data

    dev_dict = torch.load("./data/" + args.dataset + "/new_dev.dict")

    neg_num = 50  # corruptions in dict
    raw_examples = list(dev_dict.keys())
    # sample
    sample_raw_examples = [raw_examples[i] for i in range(0, len(raw_examples), 6)]
    rel_list = list(sorted(eval_dataset.rel_list))
    all_tail_ents = []
    for triplet in sample_raw_examples:
        _,_,_tail = triplet
        all_tail_ents.append(_tail)
        all_tail_ents.extend(dev_dict[triplet]["tails_corrupt"])
    all_tail_ents = list(set(all_tail_ents))  # 2612

    input_ids_list, mask_ids_list, segment_ids_list = [], [], []
    t_input_ids_list, t_mask_ids_list, t_segment_ids_list = [], [], []
    for _triplet in tqdm(sample_raw_examples):
        _head, _rel, _tail = _triplet
        all_heads = dev_dict[_triplet]["heads_corrupt"]
        all_heads.insert(0, _head)   # pos_head, + heads_corrupt
        rel_ids = str2ids(eval_dataset.rel2text[_rel])
        max_ent_len = eval_dataset.max_seq_length - 3 - len(rel_ids)
        for i, _ in enumerate(all_heads):
            head_ids = str2ids(eval_dataset.ent2text[all_heads[i]])
            head_ids = head_ids[:max_ent_len]

            # get head + rel input ids
            src_input_ids = [eval_dataset._cls_id] + head_ids + [eval_dataset._sep_id] + rel_ids + [
                eval_dataset._sep_id]
            src_mask_ids = [1] * len(src_input_ids)
            src_segment_ids = [0] * (len(head_ids) + 2) + [1] * (len(rel_ids) + 1)
            input_ids_list.append(src_input_ids)
            mask_ids_list.append(src_mask_ids)
            segment_ids_list.append(src_segment_ids)

    for tail_ent in all_tail_ents:
        max_ent_len = eval_dataset.max_seq_length - 3
        tail_ids = str2ids(eval_dataset.ent2text[tail_ent])
        tail_ids = tail_ids[:max_ent_len]
        tgt_input_ids = [eval_dataset._cls_id] + tail_ids + [eval_dataset._sep_id]
        tgt_mask_ids = [1] * len(tgt_input_ids)
        tgt_segment_ids = [0] * (len(tail_ids) + 2)
        input_ids_list.append(tgt_input_ids)
        mask_ids_list.append(tgt_mask_ids)
        segment_ids_list.append(tgt_segment_ids)

    # # padding
    max_len = max(len(_e) for _e in input_ids_list)
    assert max_len <= eval_dataset.max_seq_length
    input_ids_list = [_e + [eval_dataset._pad_id] * (max_len - len(_e)) for _e in input_ids_list]
    mask_ids_list = [_e + [0] * (max_len - len(_e)) for _e in mask_ids_list]
    segment_ids_list = [_e + [0] * (max_len - len(_e)) for _e in segment_ids_list]

    # # dataset
    enc_dataset = TensorDataset(
        torch.tensor(input_ids_list, dtype=torch.long),
        torch.tensor(mask_ids_list, dtype=torch.long),
        torch.tensor(segment_ids_list, dtype=torch.long),
    )
    enc_dataloader = DataLoader(
        enc_dataset, sampler=SequentialSampler(enc_dataset), batch_size=args.eval_batch_size * 2)

    # ------------------------  get all embeddings ----------------------------
    print("\tget all emb via model")
    embs_list = []
    for batch in tqdm(enc_dataloader, desc="embedding"):
        batch = tuple(t.to(args.device) for t in batch)
        _input_ids, _mask_ids, _segment_ids = batch
        with torch.no_grad():
            embs = model.encoder(_input_ids, attention_mask=_mask_ids, token_type_ids=_segment_ids)
            embs = embs.detach().cpu()
            embs_list.append(embs)
    emb_mat = torch.cat(embs_list, dim=0).contiguous()
    assert emb_mat.shape[0] == len(input_ids_list)

    # -------------------------  assign to ent ------------------------------

    ent_rel2emb = collections.defaultdict(dict)
    ent2emb = dict()
    ptr_row = 0
    for _triplet in sample_raw_examples:
        _head, _rel, _tail = _triplet
        all_heads = dev_dict[_triplet]["heads_corrupt"]
        for i, _ in enumerate(all_heads):
            ent_rel2emb[all_heads[i]][_rel] = emb_mat[ptr_row]
            ptr_row += 1
    for _tail in all_tail_ents:
        ent2emb[_tail] = emb_mat[ptr_row]
        ptr_row += 1


    # ---------------------------- use HIT@10 as metric -----------------------------
    ranks_left, ranks_right, ranks = [], [], []
    hits_left, hits_right, hits = [], [], []
    top_ten_hit_count = 0
    for i in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])

    for _idx_ex, _triplet in enumerate(tqdm(sample_raw_examples, desc="evaluating")):
        _head, _rel, _tail = _triplet

        head_ent_list = []
        tail_ent_list = []

        # head corrupt
        _neg_head_ents = dev_dict[_triplet]["heads_corrupt"]
        head_ent_list.append(_head)  # positive example
        head_ent_list.extend(_neg_head_ents)  # negative examples  num = 50
        tail_ent_list.extend([_tail] * (1 + len(_neg_head_ents)))
        split_idx = len(head_ent_list)

        # tail corrupt
        _neg_tail_ents = dev_dict[_triplet]["tails_corrupt"]
        head_ent_list.extend([_head] * (1 + len(_neg_tail_ents)))
        tail_ent_list.append(_tail)  # positive example
        tail_ent_list.extend(_neg_tail_ents)  # negative examples

        triplet_list = list([_h, _rel, _t] for _h, _t in zip(head_ent_list, tail_ent_list))

        # build dataset
        rep_src_list = [ent_rel2emb[_h][_rel] for _h, _rel, _ in triplet_list]
        rep_tgt_list = [ent2emb[_t] for _, _, _t in triplet_list]
        all_rep_src = torch.stack(rep_src_list, dim=0).to(args.device)
        all_rep_tgt = torch.stack(rep_tgt_list, dim=0).to(args.device)

        local_scores_list = []
        sim_batch_size = args.eval_batch_size * 8

        if args.cls_method == "dis":
            for _idx_r in range(0, all_rep_src.shape[0], sim_batch_size):
                _rep_src, _rep_tgt = all_rep_src[_idx_r: _idx_r + sim_batch_size], all_rep_tgt[
                                                                                   _idx_r: _idx_r + sim_batch_size]
                with torch.no_grad():
                    distances = model.distance_metric_fn(_rep_src, _rep_tgt)
                    local_scores = - distances
                    local_scores = local_scores.detach().cpu().numpy()
                local_scores_list.append(local_scores)
        elif args.cls_method == "cls":
            for _idx_r in range(0, all_rep_src.shape[0], sim_batch_size):
                _rep_src, _rep_tgt = all_rep_src[_idx_r: _idx_r + sim_batch_size], all_rep_tgt[
                                                                                   _idx_r: _idx_r + sim_batch_size]
                with torch.no_grad():
                    logits = model.classifier(_rep_src, _rep_tgt)
                    logits = torch.softmax(logits, dim=-1)
                    local_scores = logits.detach().cpu().numpy()[:, 1]
                local_scores_list.append(local_scores)
        elif args.cls_method == "mix":
            for _idx_r in range(0, all_rep_src.shape[0], sim_batch_size):
                _rep_src, _rep_tgt = all_rep_src[_idx_r: _idx_r + sim_batch_size], all_rep_tgt[
                                                                                   _idx_r: _idx_r + sim_batch_size]
                with torch.no_grad():
                    logits = model.classifier(_rep_src, _rep_tgt)
                    logits = torch.softmax(logits, dim=-1)[:, 1]
                    distances = model.distance_metric_fn(_rep_src, _rep_tgt)
                    if args.distance_metric == "bilinear":
                        distances = torch.softmax(distances, dim=-1)
                    local_scores = torch.div(logits, distances + 0.1)
                    local_scores = local_scores.detach().cpu().numpy()
                local_scores_list.append(local_scores)
        scores = np.concatenate(local_scores_list, axis=0)

        # left
        left_scores = scores[:split_idx]
        left_rank = safe_ranking(left_scores)
        ranks_left.append(left_rank)
        ranks.append(left_rank)

        # right
        right_scores = scores[split_idx:]
        right_rank = safe_ranking(right_scores)
        ranks_right.append(right_rank)
        ranks.append(right_rank)

        # log
        top_ten_hit_count += (int(left_rank <= 10) + int(right_rank <= 10))
        if (_idx_ex + 1) % 10 == 0:
            logger.info("hit@10 until now: {}".format(top_ten_hit_count * 1.0 / len(ranks)))
            logger.info('mean rank until now: {}'.format(np.mean(ranks)))

        # hits
        for hits_level in range(10):
            if left_rank <= hits_level + 1:
                hits[hits_level].append(1.0)
                hits_left[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)
                hits_left[hits_level].append(0.0)

            if right_rank <= hits_level + 1:
                hits[hits_level].append(1.0)
                hits_right[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)
                hits_right[hits_level].append(0.0)

    for i in [0, 2, 9]:
        logger.info('Hits left @{0}: {1}'.format(i + 1, np.mean(hits_left[i])))
        logger.info('Hits right @{0}: {1}'.format(i + 1, np.mean(hits_right[i])))
        logger.info('Hits @{0}: {1}'.format(i + 1, np.mean(hits[i])))
    logger.info('Mean rank left: {0}'.format(np.mean(ranks_left)))
    logger.info('Mean rank right: {0}'.format(np.mean(ranks_right)))
    logger.info('Mean rank: {0}'.format(np.mean(ranks)))
    logger.info('Mean reciprocal rank left: {0}'.format(np.mean(1. / np.array(ranks_left))))
    logger.info('Mean reciprocal rank right: {0}'.format(np.mean(1. / np.array(ranks_right))))
    logger.info('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))

    output_eval_file = os.path.join(args.output_dir, file_prefix + "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logging.info("***** Eval results at {}*****".format(global_step))
        writer.write("***** Eval results at {}*****\n".format(global_step))
        for i in [0, 2, 9]:
            writer.write('Hits left @{0}: {1}\n'.format(i + 1, np.mean(hits_left[i])))
            writer.write('Hits right @{0}: {1}\n'.format(i + 1, np.mean(hits_right[i])))
            writer.write('Hits @{0}: {1}\n'.format(i + 1, np.mean(hits[i])))
        writer.write('Mean rank left: {0}\n'.format(np.mean(ranks_left)))
        writer.write('Mean rank right: {0}\n'.format(np.mean(ranks_right)))
        writer.write('Mean rank: {0}\n'.format(np.mean(ranks)))
        writer.write('Mean reciprocal rank left: {0}\n'.format(np.mean(1. / np.array(ranks_left))))
        writer.write('Mean reciprocal rank right: {0}\n'.format(np.mean(1. / np.array(ranks_right))))
        writer.write('Mean reciprocal rank: {0}\n'.format(np.mean(1. / np.array(ranks))))
        writer.write("\n")

    return np.mean(hits[9])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_class", default="roberta", type=str,
                        help="model class, one of [bert, roberta]")
    parser.add_argument("--dataset", type=str, default="WN18RR")
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--neg_weights", default=None, type=str)

    parser.add_argument("--distance_metric", default="euclidean", type=str)   # 默认距离度量为mlp
    parser.add_argument("--hinge_loss_margin", default=1., type=float)
    parser.add_argument("--pos_weight", default=1, type=float)
    parser.add_argument("--loss_weight", default=1, type=float)
    parser.add_argument("--cls_loss_weight", default=1, type=float)
    parser.add_argument("--cls_method", default="cls", type=str)

    # extra parameters for prediction
    parser.add_argument("--no_verbose", action="store_true")
    parser.add_argument("--collect_prediction", action="store_true")
    parser.add_argument("--prediction_part", default="0,1", type=str)

    # parameter for negative sampling
    parser.add_argument("--type_cons_neg_sample", action="store_true")
    parser.add_argument("--type_cons_ratio", default=0, type=float)




    ## Other parameters
    define_hparams_training(parser)
    args = parser.parse_args()

    # setup
    setup_prerequisite(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab



    if args.model_class == "bert":
        config_class = BertConfig
        tokenizer_class = BertTokenizer
        model_class = BertForPairScoring
    elif args.model_class == "roberta":
        config_class = RobertaConfig
        tokenizer_class = RobertaTokenizer
        model_class = RobertaForPairScoring
    else:
        raise KeyError(args.model_class)

    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
    )
    config.distance_metric = args.distance_metric
    config.hinge_loss_margin = args.hinge_loss_margin
    config.pos_weight = args.pos_weight
    config.loss_weight = args.loss_weight
    config.cls_loss_weight = args.cls_loss_weight

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Dataset
    neg_weights = [1., 1., 0.] if args.neg_weights is None else [float(_e) for _e in args.neg_weights.split(",")]
    assert len(neg_weights) == 3 and sum(neg_weights) > 0

    train_dataset = DatasetForPairwiseRankingLP(
        args.dataset, "train", None, "./data/",
        args.model_class, tokenizer, args.do_lower_case,
        args.max_seq_length,  neg_times=5 ,neg_weights=neg_weights,
        type_cons_neg_sample=args.type_cons_neg_sample, type_cons_ratio=args.type_cons_ratio
    )


    dev_dataset = DatasetForPairwiseRankingLP(
        args.dataset, "dev", None, "./data/",
        args.model_class, tokenizer, args.do_lower_case,
        args.max_seq_length,
    )
    test_dataset = DatasetForPairwiseRankingLP(
        args.dataset, "test", None, "./data/",
        args.model_class, tokenizer, args.do_lower_case,
        args.max_seq_length,
    )

    if args.do_train:
        train(args, train_dataset, model, tokenizer, eval_dataset=dev_dataset, eval_fn=evaluate_pairwise_ranking)

    if args.fp16:
        model = setup_eval_model_for_fp16(args, model)

    if args.do_prediction:
        dataset_list = [train_dataset, dev_dataset, test_dataset]
        if args.dataset == "NELL_standard":
            predict_NELL(args, test_dataset.raw_examples, dataset_list, model, verbose=True)
        else:
            tuple_ranks = predict(
                args, test_dataset.raw_examples, dataset_list, model, verbose=True)
            output_str = calculate_metrics_for_link_prediction(tuple_ranks, verbose=True)
            save_json(tuple_ranks, join(args.output_dir, "tuple_ranks.json"))
            with open(join(args.output_dir, "link_prediction_metrics.txt"), "w", encoding="utf-8") as fp:
                fp.write(output_str)

if __name__ == '__main__':
    main()

