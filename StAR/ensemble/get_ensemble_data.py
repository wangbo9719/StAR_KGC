from peach.help import *
import argparse
import collections
import os
import random
import torch
from os.path import join
from peach.common import file_exists, dir_exists, save_json, load_json, StAR_FILE_PATH
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer, RobertaConfig, RobertaTokenizer
from kbc.kb_dataset import KbDataset
from kbc.models import BertForPairScoring, RobertaForPairScoring
from kbc.metric import calculate_metrics_for_link_prediction, safe_ranking
import numpy as np

def safe_ranking2(_scores, _pos_idx):
    pos_score = _scores[_pos_idx]  # []
    same_score_loc = np.where(_scores == pos_score)[0]
    assert same_score_loc.size > 0
    rdm_pos_loc = same_score_loc[random.randint(0, same_score_loc.shape[0]-1)]
    _sort_idxs = np.argsort(-_scores)
    _rank = np.where(_sort_idxs == rdm_pos_loc)[0][0] + 1

    return _rank

class DatasetForPairwiseRankingLP(KbDataset):
    def __init__(self, *arg, **kwargs):
        super(DatasetForPairwiseRankingLP, self).__init__(*arg, **kwargs)
        self.ent2idx = self.get_dict1(os.path.join(self.data_dir, "entities.dict"))
        self.rel2idx = self.get_dict1(os.path.join(self.data_dir, "relations.dict"))
        self.id2ent = self.get_dict2(os.path.join(self.data_dir, "entities.dict"))
        self.id2rel = self.get_dict2(os.path.join(self.data_dir, "relations.dict"))
        self.id2ent_list = list(self.id2ent.values())
        self.id2rel_list = list(self.id2rel.values())


    def get_dict1(self, data_path):
        with open(data_path) as fin:
            txt2id = dict()
            for line in fin:
                id, txt = line.strip().split('\t')
                txt2id[txt] = int(id)
        return txt2id

    def get_dict2(self, data_path):
        with open(data_path) as fin:
            id2txt = dict()
            for line in fin:
                id, txt = line.strip().split('\t')
                id2txt[int(id)] = txt
        return id2txt

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
                # for _tensor in _tensors:
                #     _pad_shape = list(_tensor.size())
                #     _pad_shape[1] = _max_len_last_dim - _tensor.size(1)
                #     _pad_tensor = torch.full(_pad_shape, padding_value, device=_tensor.device, dtype=_tensor.dtype)
                #     _new_tensor = torch.cat([_tensor, _pad_tensor], dim=1)
                #     _new_tensors.append(_new_tensor)
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

def get_emb_mat(save_dir, save_path, ent_list, rel_list, dataset, args, model=None, verbose=True):
    if dir_exists(save_dir) and file_exists(save_path):
        logging.info("load from file")
        emb_mat = torch.load(save_path)
    else:
        logging.info("get all ids")
        input_ids_list, mask_ids_list, segment_ids_list = [], [], []
        for _ent in tqdm(ent_list):
            for _idx_r, _rel in enumerate(rel_list):
                head_ids, rel_ids, tail_ids = dataset.convert_raw_example_to_features(
                    [_ent, _rel, _ent], method="4")
                head_ids, rel_ids, tail_ids = head_ids[1:-1], rel_ids[1:-1], tail_ids[1:-1]
                # truncate
                max_ent_len = dataset.max_seq_length - 3 - len(rel_ids)
                head_ids = head_ids[:max_ent_len]
                tail_ids = tail_ids[:max_ent_len]

                src_input_ids = [dataset._cls_id] + head_ids + [dataset._sep_id] + rel_ids + [
                    dataset._sep_id]
                src_mask_ids = [1] * len(src_input_ids)
                src_segment_ids = [0] * (len(head_ids) + 2) + [1] * (len(rel_ids) + 1)

                if _idx_r == 0:
                    tgt_input_ids = [dataset._cls_id] + tail_ids + [dataset._sep_id]
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
        assert max_len <= dataset.max_seq_length
        input_ids_list = [_e + [dataset._pad_id] * (max_len - len(_e)) for _e in input_ids_list]
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
        logging.info("get all emb via model")
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
    return emb_mat

def assign_emb2elements(ent_list, rel_list, emb_mat):
    assert len(ent_list) * (1 + len(rel_list)) == emb_mat.shape[0]
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
    return ent2emb, ent_rel2emb

def get_scores(args, raw_examples, dataset, model, data_type=None, verbose=True):

    save_dir = args.model_name_or_path if dir_exists(args.model_name_or_path) else args.output_dir
    save_path = os.path.join(save_dir, "saved_emb_mat.np")
    head_scores_path = os.path.join(save_dir, data_type+"_head_full_scores.list")
    tail_scores_path = os.path.join(save_dir, data_type+"_tail_full_scores.list")
    if file_exists(head_scores_path) and file_exists(tail_scores_path):
        logging.info("Load head and tail mode scores")
        head_scores = torch.load(head_scores_path)
        tail_scores = torch.load(tail_scores_path)
        return head_scores, tail_scores
    model.eval()
    ent_list = list(sorted(list(dataset.ent2idx.keys())))
    rel_list = list(sorted(list(dataset.rel2idx.keys())))
    logging.info("Load all embeddings")
    emb_mat = get_emb_mat(save_dir, save_path, ent_list, rel_list, dataset, args, model)
    ent2emb, ent_rel2emb = assign_emb2elements(ent_list, rel_list, emb_mat)
    # ========== get ranked logits score ==========
    head_scores = []
    tail_scores = []
    split_idx = len(dataset.id2ent_list)
    id2ent_list = dataset.id2ent_list
    for _idx_ex, _triplet in enumerate(tqdm(raw_examples, desc="get_" + data_type + "_scores")):
        _head, _rel, _tail = _triplet

        head_ent_list = id2ent_list
        tail_ent_list = [_tail] * split_idx

        head_ent_list = head_ent_list + [_head]*split_idx
        tail_ent_list = tail_ent_list + id2ent_list

        triplet_list = list([_h, _rel, _t] for _h, _t in zip(head_ent_list, tail_ent_list))

        # build dataset
        rep_src_list = [ent_rel2emb[_h][_rel] for _h, _rel, _ in triplet_list]
        rep_tgt_list = [ent2emb[_t] for _, _rel, _t in triplet_list]
        all_rep_src = torch.stack(rep_src_list, dim=0).to(args.device)
        all_rep_tgt = torch.stack(rep_tgt_list, dim=0).to(args.device)


        local_logits_list = []
        sim_batch_size = args.eval_batch_size * 8
        for _idx_r in range(0, all_rep_src.shape[0], sim_batch_size):
            _rep_src, _rep_tgt = all_rep_src[_idx_r: _idx_r + sim_batch_size], all_rep_tgt[
                                                                               _idx_r: _idx_r + sim_batch_size]

            with torch.no_grad():
                logits = model.classifier(_rep_src, _rep_tgt)
                logits = torch.softmax(logits, dim=-1)
                local_scores = logits.detach().cpu().numpy()[:, 1]
            local_logits_list.append(local_scores)

        sample_logits_list = np.concatenate(local_logits_list, axis=0)
        head_scores.append([np.array(dataset.ent2idx[_head]), sample_logits_list[:split_idx]])
        tail_scores.append([np.array(dataset.ent2idx[_tail]), sample_logits_list[split_idx:]])

    if dir_exists(save_dir):
        torch.save(head_scores, head_scores_path)
        torch.save(tail_scores, tail_scores_path)
    logging.info("Get scores finished")
    return head_scores, tail_scores

def get_model_dataset(args, model, dataset_list, data_type='test', top_n=1000, verbose=True):
    model.eval()
    if data_type == 'train':
        raw_examples = dataset_list[0].raw_examples
    if data_type == 'dev':
        raw_examples = dataset_list[1].raw_examples
    elif data_type == 'test':
        raw_examples = dataset_list[2].raw_examples
    dataset = dataset_list[2]
    # get the last one (i.e., test) to make use if its useful functions and data
    ent_list = list(sorted(list(dataset.ent2idx.keys())))
    rel_list = list(sorted(list(dataset.rel2idx.keys())))
    g_subj2objs = collections.defaultdict(lambda: collections.defaultdict(set))
    g_obj2subjs = collections.defaultdict(lambda: collections.defaultdict(set))

    # prepare to remove the true triples
    for _ds in dataset_list:
        for _raw_ex in _ds.raw_examples:
            _head, _rel, _tail = _raw_ex
            g_subj2objs[_head][_rel].add(_tail)
            g_obj2subjs[_tail][_rel].add(_head)

    save_dir = args.model_name_or_path if dir_exists(args.model_name_or_path) else args.output_dir
    save_path = os.path.join(save_dir, "saved_emb_mat.np")
    emb_mat = get_emb_mat(save_dir, save_path, ent_list, rel_list, dataset, args, model)
    ent2emb, ent_rel2emb = assign_emb2elements(ent_list, rel_list, emb_mat)
    if data_type == 'train':
        head_scores_path = os.path.join(save_dir, data_type+"_head_topN_scores.list")  # "_head_scores.list"
        tail_scores_path = os.path.join(save_dir, data_type+"_tail_topN_scores.list")  # "_tail_scores.list"
    elif data_type == 'dev' or data_type =='test':
        full_head_scores_path = os.path.join(save_dir, data_type + "_head_full_scores.list")
        full_tail_scores_path = os.path.join(save_dir, data_type + "_tail_full_scores.list")

    # ========== get ranked logits score and corresponding index ==========
    full_head_scores = []
    full_tail_scores = []
    head_scores = []
    tail_scores = []
    split_idx = len(dataset.id2ent_list)
    id2ent_list = dataset.id2ent_list
    head_triple_idx_list = []
    tail_triple_idx_list = []
    for _idx_ex, _triplet in enumerate(tqdm(raw_examples, desc="Get_" + data_type + "_datasets")):
        _head, _rel, _tail = _triplet

        head_ent_list = id2ent_list
        tail_ent_list = [_tail] * split_idx

        head_ent_list = head_ent_list + [_head]*split_idx
        tail_ent_list = tail_ent_list + id2ent_list

        triplet_list = list([_h, _rel, _t] for _h, _t in zip(head_ent_list, tail_ent_list))

        # build dataset
        rep_src_list = [ent_rel2emb[_h][_rel] for _h, _rel, _ in triplet_list]
        rep_tgt_list = [ent2emb[_t] for _, _rel, _t in triplet_list]
        all_rep_src = torch.stack(rep_src_list, dim=0).to(args.device)
        all_rep_tgt = torch.stack(rep_tgt_list, dim=0).to(args.device)


        local_logits_list = []
        sim_batch_size = args.eval_batch_size * 8
        for _idx_r in range(0, all_rep_src.shape[0], sim_batch_size):
            _rep_src, _rep_tgt = all_rep_src[_idx_r: _idx_r + sim_batch_size], all_rep_tgt[
                                                                               _idx_r: _idx_r + sim_batch_size]

            with torch.no_grad():
                logits = model.classifier(_rep_src, _rep_tgt)
                logits = torch.softmax(logits, dim=-1)
                local_scores = logits.detach().cpu().numpy()[:, 1]
            local_logits_list.append(local_scores)

        sample_logits_list = np.concatenate(local_logits_list, axis=0)
        full_head_scores.append([np.array(dataset.ent2idx[_head]), sample_logits_list[:split_idx]])
        full_tail_scores.append([np.array(dataset.ent2idx[_tail]), sample_logits_list[split_idx:]])

        head_logits_list = sample_logits_list[:split_idx]
        tail_logits_list = sample_logits_list[split_idx:]

        pos_head_idx, pos_tail_idx = dataset.ent2idx[_head], dataset.ent2idx[_tail]

        head_pos_score = head_logits_list[pos_head_idx]
        head_sort_idxs = np.argsort(- head_logits_list)
        head_pos_rank = np.where(head_sort_idxs == pos_head_idx)[0][0] + 1
        head_top_ranked_score = head_logits_list[head_sort_idxs[:top_n]]
        head_top_ranked_idx = head_sort_idxs[:top_n]

        tail_pos_score = tail_logits_list[pos_tail_idx]
        tail_sort_idxs = np.argsort(- tail_logits_list)
        tail_pos_rank = np.where(tail_sort_idxs == pos_tail_idx)[0][0] + 1
        tail_top_ranked_score = tail_logits_list[tail_sort_idxs[:top_n]]
        tail_top_ranked_idx = tail_sort_idxs[:top_n]

        # For each triple head/tail info: [[pos_idx, pos_score, pos_ranking],[top-N score], [top-N idx]
        if head_pos_rank <= top_n:
            head_scores.append([head_pos_rank - 1, head_top_ranked_score, head_top_ranked_idx.astype(np.int32)])
            head_triple_idx_list.append(_idx_ex)
        if tail_pos_rank <= top_n:
            tail_scores.append([tail_pos_rank - 1, tail_top_ranked_score, tail_top_ranked_idx.astype(np.int32)])
            tail_triple_idx_list.append(_idx_ex)
    if dir_exists(save_dir):
        if data_type == 'train':
            torch.save(head_scores, head_scores_path)
            torch.save(tail_scores, tail_scores_path)
            torch.save(head_triple_idx_list, join(save_dir, data_type + '_head_triple_idx.list'))
            torch.save(tail_triple_idx_list, join(save_dir, data_type + '_tail_triple_idx.list'))
        elif data_type == 'dev' or data_type =='test':
            torch.save(full_head_scores, full_head_scores_path)
            torch.save(full_tail_scores, full_tail_scores_path)


    print("Get scores and datasets finished")

    emb_list = []
    for _i in range(len(ent_list)):
        id2ent = dataset.id2ent
        emb_list.append(ent2emb[id2ent[_i]])
    emb_list = torch.stack(emb_list, dim=0)
    if dir_exists(save_dir):
        torch.save(emb_list, os.path.join(save_dir, "ent_emb.pkl"))
    logging.info("***** Save all entities embedding finished. *****")

def get_similarity(args, test_dataset, top_n):
    logging.info("***** Get Similarity *****")

    ent_list = list(sorted(list(test_dataset.ent2idx.keys())))
    rel_list = list(sorted(list(test_dataset.rel2idx.keys())))
    id2ent = test_dataset.id2ent

    # ========= get all embeddings ==========
    save_dir = args.model_name_or_path if dir_exists(args.model_name_or_path) else args.output_dir
    save_path = os.path.join(save_dir, "saved_emb_mat.np")

    emb_mat = get_emb_mat(save_dir, save_path, ent_list, rel_list, test_dataset, args)
    ent2emb, ent_rel2emb = assign_emb2elements(ent_list, rel_list, emb_mat)

    emb_list = []
    for i in range(len(ent_list)):
        emb_list.append(ent2emb[id2ent[i]])
    emb_list = torch.stack(emb_list, dim=0).to(args.device)

    similarity_score_mtx = []
    # similarity_index_mtx = []
    # if args.get_cosine_similarity:
    for i in tqdm(range(len(ent_list))):
        line_similarity = torch.cosine_similarity(torch.stack([emb_list[i]]*len(emb_list)), emb_list, dim=-1).detach().cpu().numpy()
        line_simi_index = np.argsort(-line_similarity)  # np.argsort sorts elements from small to large
        line_simi_score = line_similarity[line_simi_index]
        # similarity_index_mtx.append(line_simi_index[:top_n])
        similarity_score_mtx.append(line_simi_score[:top_n])
    # similarity_index_mtx = np.array(similarity_index_mtx)
    similarity_score_mtx = np.array(similarity_score_mtx)
    if dir_exists(save_dir):
            # np.save(os.path.join(save_dir, "similarity_index_mtx"), similarity_index_mtx)
            np.save(os.path.join(save_dir, "similarity_score_mtx"), similarity_score_mtx)
    # else:
    #     raise NotImplementedError

    return similarity_score_mtx

def get_ent_emb(args, test_dataset):
    logging.info("***** Get all entities embedding *****")

    ent_list = list(sorted(list(test_dataset.ent2idx.keys())))
    rel_list = list(sorted(list(test_dataset.rel2idx.keys())))
    id2ent = test_dataset.id2ent

    # ========= get all embeddings ==========
    save_dir = args.model_name_or_path if dir_exists(args.model_name_or_path) else args.output_dir
    save_path = os.path.join(save_dir, "saved_emb_mat.np")
    emb_mat = get_emb_mat(save_dir, save_path, ent_list, rel_list, test_dataset, args)
    ent2emb, ent_rel2emb = assign_emb2elements(ent_list, rel_list, emb_mat)

    emb_list = []
    for i in range(len(ent_list)):
        emb_list.append(ent2emb[id2ent[i]])
    emb_list = torch.stack(emb_list, dim=0)

    if dir_exists(save_dir):
        torch.save(emb_list,os.path.join(save_dir, "ent_emb.pkl"))
    logging.info("***** Finished *****")

def collect_case(args, raw_examples, dataset_list, model, verbose=True):
    logging.info("***** Running Getting Cases*****")

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
    save_dir = args.model_name_or_path if dir_exists(args.model_name_or_path) else args.output_dir
    save_path = os.path.join(save_dir, "saved_emb_mat.np")
    case_save_path = os.path.join(save_dir, "cases_alone.txt")
    dict_save_path = join(save_dir, 'cases_alone.dict')
    emb_mat = get_emb_mat(save_dir, save_path, ent_list, rel_list, standard_dataset, args, model)
    ent2emb, ent_rel2emb = assign_emb2elements(ent_list, rel_list, emb_mat)
    # ========== get logits and distances ==========

    results_dict = collections.defaultdict(dict)
    ent2text_dict = dataset_list[0].ent2text
    for ent_id, ent_text in ent2text_dict.items():
        #################NOTE#########################
        ent2text_dict[ent_id] = ent_text.split(",")[0]
        # ent2text_dict[ent_id] = ent_text
    for _idx_ex, _triplet in enumerate(tqdm(raw_examples, desc="get cases")):
        _head, _rel, _tail = _triplet

        head_ent_list = []
        tail_ent_list = []

        # head corrupt
        _pos_head_ents = g_obj2subjs[_tail][_rel]
        _neg_head_ents = ents - _pos_head_ents
        head_ent_list.append(_head)  # positive example
        head_ent_list.extend(_neg_head_ents)  # negative examples
        tail_ent_list.extend([_tail] * (1 + len(_neg_head_ents)))
        split_idx = len(head_ent_list)

        # tail corrupt
        _pos_tail_ents = g_subj2objs[_head][_rel]
        _neg_tail_ents = ents - _pos_tail_ents
        head_ent_list.extend([_head] * (1 + len(_neg_tail_ents)))
        tail_ent_list.append(_tail)  # positive example
        tail_ent_list.extend(_neg_tail_ents)  # negative examples

        triplet_list = list([_h, _rel, _t] for _h, _t in zip(head_ent_list, tail_ent_list))
        # all triples to be verified for a test sample

        # build dataset
        rep_src_list = [ent_rel2emb[_h][_rel] for _h, _rel, _ in triplet_list]
        rep_tgt_list = [ent2emb[_t] for _, _rel, _t in triplet_list]
        all_rep_src = torch.stack(rep_src_list, dim=0).to(args.device)
        all_rep_tgt = torch.stack(rep_tgt_list, dim=0).to(args.device)

        local_logits_list = []
        local_distances_list = []
        sim_batch_size = args.eval_batch_size * 8
        for _idx_r in range(0, all_rep_src.shape[0], sim_batch_size):
            _rep_src, _rep_tgt = all_rep_src[_idx_r: _idx_r + sim_batch_size], all_rep_tgt[
                                                                               _idx_r: _idx_r + sim_batch_size]

            with torch.no_grad():
                logits = model.classifier(_rep_src, _rep_tgt)
                distances = model.distance_metric_fn(_rep_src, _rep_tgt)
                logits = logits.detach().cpu().numpy()
                distances = distances.detach().cpu().numpy()
            local_logits_list.append(logits)
            local_distances_list.append(distances)

        sample_logits_list = np.concatenate(local_logits_list, axis=0)
        sample_distances_list = np.concatenate(local_distances_list, axis=0)
        sample_logits_list = torch.from_numpy(sample_logits_list)
        scores = torch.softmax(sample_logits_list,dim=-1)[:,1]
        scores = scores.detach().cpu().numpy()

        # left
        left_scores = scores[:split_idx]
        left_sort_idxs = np.argsort(-left_scores)
        left_rank = np.where(left_sort_idxs == 0)[0][0] + 1
        left_sort_idxs = left_sort_idxs[:20]
        left_rk_inf = []
        left_rk_inf.append([ent2text_dict[_head],left_rank,left_scores[0]])
        for i, idx in enumerate(left_sort_idxs):
            corrupt_rank = np.where(left_sort_idxs == idx)[0][0] + 1
            corrupt_text = ent2text_dict[head_ent_list[idx]]
            left_rk_inf.append([corrupt_text,corrupt_rank,left_scores[idx]])

        # right
        right_scores = scores[split_idx:]
        right_sort_idxs = np.argsort(-right_scores)
        right_rank = np.where(right_sort_idxs == 0)[0][0] + 1
        right_sort_idxs = right_sort_idxs[:20]
        right_rk_inf = []
        right_rk_inf.append([ent2text_dict[_tail], right_rank, right_scores[0]])
        for i, idx in enumerate(right_sort_idxs):
            corrupt_rank = np.where(right_sort_idxs == idx)[0][0] + 1
            corrupt_text = ent2text_dict[tail_ent_list[split_idx + idx]]
            right_rk_inf.append([corrupt_text, corrupt_rank, right_scores[idx]])



        _head_text = ent2text_dict[_head]
        _tail_text = ent2text_dict[_tail]
        _triplet_text = tuple([_head_text,_rel,_tail_text])
        # add infor to results_dict  {triples:{head:[pos ranking，others[entity_text, score]], tail:[]},...}
        results_dict[_triplet_text] = {"head":left_rk_inf, "tail":right_rk_inf}

        with open(case_save_path, 'a', encoding='utf-8') as f:
            f.write(str([_head_text,_rel,_tail_text]) + '\n')
            f.write("head:" + str(left_rk_inf) + '\n')
            f.write("tail:" + str(right_rk_inf) + '\n\n')

    logging.info("Get cases in text finished")
    if dir_exists(save_dir):
        torch.save(results_dict, dict_save_path)
    logging.info("Get cases in dict finished")

def get_improve_cases(args):
    stelp_dict = torch.load(join(args.context_score_path, 'cases_alone.dict'))
    rotate_dict = torch.load(join(args.translation_score_path,'RotatE_case_alone.dict'))
    add_dict = torch.load(join(args.context_score_path, 'add_cases.dict'))
    improve_dict = dict()
    for key in stelp_dict:
        _h, _r, _t = key
        stelp_rank = stelp_dict[key]['tail'][0][1]
        _r = _r.replace('_',' ')[1:]
        rotate_rank = rotate_dict[(_h,_r,_t)]['tail'][0][1]
        add_rank = add_dict[key]['tail'][0][1]
        if stelp_rank > add_rank and rotate_rank > add_rank:
            if add_dict[key]['tail'][0][1] <= 10:
                improve_dict[key] = [add_rank, stelp_rank, rotate_rank]

    torch.save(improve_dict, join(args.context_score_path, 'improve_tail_case.dict'))
    with open(join(args.context_score_path, 'improve_tail_case.txt'), 'a', encoding='utf-8') as f:
        for key in improve_dict:
            f.write(str(key) + ': ')
            f.write(str(improve_dict[key]) + '\n')


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

    parser.add_argument("--distance_metric", default="euclidean", type=str)
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

    # parameter for ensemble
    parser.add_argument("--get_scores", action="store_true")
    parser.add_argument("--get_cosine_similarity", action="store_true")
    parser.add_argument("--context_score_path", default=None, type=str)
    parser.add_argument("--translation_score_path", default=None, type=str)
    parser.add_argument("--get_improve_cases", action="store_true")
    parser.add_argument("--collect_case", action="store_true")

    parser.add_argument("--simple_add_scores", action="store_true") # check

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

    if StAR_FILE_PATH is None:
        print("Please replace StAR_FILE_PATH in ./StAR/peach/common.py with your own path to run the code.")
        return
    
    train_dataset = DatasetForPairwiseRankingLP(
        args.dataset, "train", None, StAR_FILE_PATH+"/StAR/data/",
        args.model_class, tokenizer, args.do_lower_case,
        args.max_seq_length,  neg_times=5 ,neg_weights=neg_weights,
        type_cons_neg_sample=args.type_cons_neg_sample, type_cons_ratio=args.type_cons_ratio
    )
    dev_dataset = DatasetForPairwiseRankingLP(
        args.dataset, "dev", None, StAR_FILE_PATH+"/StAR/data/",
        args.model_class, tokenizer, args.do_lower_case,
        args.max_seq_length,
    )
    test_dataset = DatasetForPairwiseRankingLP(
        args.dataset, "test", None, StAR_FILE_PATH+"/StAR/data/",
        args.model_class, tokenizer, args.do_lower_case,
        args.max_seq_length,
    )
    dataset_list = [train_dataset, dev_dataset, test_dataset]

    # Get the scores of development and test dataset predicted by StAR and corresponding datasets of ensemble models
    for type in ['train','dev','test']:
        get_model_dataset(args, model, dataset_list, data_type=type)
    # Get the largest 100 cosine similarities between each candidate and all entities in entity set
    get_similarity(args, test_dataset, 100)

    if args.collect_case:
        collect_case(args, test_dataset.raw_examples, dataset_list, model)
    if args.get_improve_cases:
        get_improve_cases(args)
        logging.info("save finished!")
    if args.collect_prediction:
        tuple_ranks = load_json(join(args.model_name_or_path, "tuple_ranks.json"))
        calculate_metrics_for_link_prediction(tuple_ranks, verbose=True)


if __name__ == '__main__':
    main()


