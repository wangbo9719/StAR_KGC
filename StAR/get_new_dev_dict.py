"""
Use the way of h+r == t paradigm
"""

from peach.help import *
from torch.utils.data import Dataset, TensorDataset, DataLoader
from os.path import join
from peach.common import save_json, load_json, save_list_to_file, load_list_from_file, file_exists, dir_exists
from tqdm import tqdm, trange
import collections
import logging as logger
import numpy as np
import random
import torch
import torch.nn as nn
from transformers import BertConfig, BertTokenizer, BertPreTrainedModel, BertModel
from transformers import RobertaConfig, RobertaTokenizer
from kbc.models import BertForPairCls, RobertaForPairCls
from tensorboardX import SummaryWriter
import argparse
from kbc.kb_dataset import KbDataset
from kbc.metric import calculate_metrics_for_link_prediction, safe_ranking
from kbc.utils_fn import train
from peach.mutli_proc import multiprocessing_map, combine_from_lists, split_to_lists

def get_triplet_candidate(_scores):
    pos_score = _scores[0]  # []
    same_score_loc = np.where(_scores == pos_score)[0]
    assert same_score_loc.size > 0 and same_score_loc[0] == 0
    rdm_pos_loc = same_score_loc[random.randint(0, same_score_loc.shape[0] - 1)]
    _sort_idxs = np.argsort(-_scores)
    _rank = np.where(_sort_idxs == rdm_pos_loc)[0][0] + 1
    _sort_idxs = list(_sort_idxs)
    if 0 in _sort_idxs[:50]:
        _sort_idxs.remove(0)
    return _sort_idxs[:50]

def get_dev_candidate(args, raw_examples, dataset_list, model, verbose=True):
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

    save_dir = args.model_name_or_path
    save_path = os.path.join(args.model_name_or_path, "saved_dev_emb_mat.np")
    new_dev_path = os.path.join("./data/"+args.dataset+"/new_dev.dict")

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
            enc_dataset, sampler=SequentialSampler(enc_dataset), batch_size=args.eval_batch_size * 2)
        print("\tget all emb via model")
        embs_list = []
        rep_torch_dtype = None
        for batch in tqdm(enc_dataloader, desc="entity embedding", disable=(not verbose)):
            batch = tuple(t.to(args.device) for t in batch)
            _input_ids, _mask_ids, _segment_ids = batch
            with torch.no_grad():
                embs = model.encoder(_input_ids, attention_mask=_mask_ids, token_type_ids=_segment_ids)
                if rep_torch_dtype is None:
                    rep_torch_dtype = embs.dtype
                embs = embs.detach().cpu()
                embs_list.append(embs)

        emb_mat = torch.cat(embs_list, dim=0).contiguous()
        assert emb_mat.shape[0] == len(input_ids_list)
        # save emb_mat
        if dir_exists(save_dir):
            torch.save(emb_mat, save_path)

    # # assign to ent
    assert len(ent_list) * (1 + len(rel_list)) == emb_mat.shape[0]

    ent_rel2emb = collections.defaultdict(dict)   # ent + r  (h + r
    ent2emb = dict()    # ent  (t

    ptr_row = 0
    for _ent in ent_list:
        for _idx_r, _rel in enumerate(rel_list):
            if _idx_r == 0:
                ent2emb[_ent] = emb_mat[ptr_row]
                ptr_row += 1
            ent_rel2emb[_ent][_rel] = emb_mat[ptr_row]
            ptr_row += 1

    # ==========  get candidates ==========

    # * begin to get hit
    new_dev_dict = collections.defaultdict(dict)

    for _idx_ex, _triplet in enumerate(tqdm(raw_examples, desc="get candidates")):
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

        # build dataset
        rep_src_list = [ent_rel2emb[_h][_rel] for _h, _rel, _ in triplet_list]
        rep_tgt_list = [ent2emb[_t] for _, _, _t in triplet_list]
        all_rep_src = torch.stack(rep_src_list, dim=0).to(args.device)
        all_rep_tgt = torch.stack(rep_tgt_list, dim=0).to(args.device)

        local_scores_list = []
        sim_batch_size = args.eval_batch_size * 8
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
        heads_corrupt_idx = get_triplet_candidate(left_scores)
        heads_corrupt = [head_ent_list[i] for i in heads_corrupt_idx]

        right_scores = scores[split_idx:]
        tails_corrupt_idx = get_triplet_candidate(right_scores)
        tails_corrupt = [tail_ent_list[i+split_idx] for i in tails_corrupt_idx]

        new_dev_dict[tuple(_triplet)]["heads_corrupt"] = heads_corrupt
        new_dev_dict[tuple(_triplet)]["tails_corrupt"] = tails_corrupt
    torch.save(new_dev_dict, new_dev_path)


class LinkPredictionPairDataset(KbDataset):
    def __init__(self, *arg, **kwargs):
        super(LinkPredictionPairDataset, self).__init__(*arg, **kwargs)

    def __getitem__(self, item):
        raw_data, label = super(LinkPredictionPairDataset, self).__getitem__(item)
        head_ids, rel_ids, tail_ids = self.convert_raw_example_to_features(raw_data, method="4")
        head_ids, rel_ids, tail_ids = head_ids[1:-1], rel_ids[1:-1], tail_ids[1:-1]
        # truncate
        max_ent_len = self.max_seq_length - 3 - len(rel_ids)
        head_ids = head_ids[:max_ent_len]
        tail_ids = tail_ids[:max_ent_len]

        src_input_ids = [self._cls_id] + head_ids + [self._sep_id] + rel_ids + [self._sep_id]
        src_mask_ids = [1] * len(src_input_ids)
        src_segment_ids = [0] * (len(head_ids) + 2) + [1] * (len(rel_ids) + 1)

        tgt_input_ids = [self._cls_id] + tail_ids + [self._sep_id]
        tgt_mask_ids = [1] * len(tgt_input_ids)
        tgt_segment_ids = [0] * (len(tail_ids) + 2)

        # return
        return (
            torch.tensor(src_input_ids, dtype=torch.long),
            torch.tensor(src_mask_ids, dtype=torch.long),
            torch.tensor(src_segment_ids, dtype=torch.long),
            torch.tensor(tgt_input_ids, dtype=torch.long),
            torch.tensor(tgt_mask_ids, dtype=torch.long),
            torch.tensor(tgt_segment_ids, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    @classmethod
    def batch2feed_dict(cls, batch, data_format=None):
        inputs = {
            'src_input_ids': batch[0],  # bs, sl
            'src_attention_mask': batch[1],  #
            'src_token_type_ids': batch[2],  #
            'tgt_input_ids': batch[3],  # bs, sl
            'tgt_attention_mask': batch[4],  #
            'tgt_token_type_ids': batch[5],  #
            "labels": batch[-1],  #
        }
        return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_class", default="roberta", type=str,
                        help="model class, one of [bert, roberta]")
    parser.add_argument("--dataset", type=str, default="wn18rr")
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--neg_weights", default=None, type=str)

    # extra parameters for prediction
    parser.add_argument("--no_verbose", action="store_true")
    parser.add_argument("--collect_prediction", action="store_true")
    parser.add_argument("--prediction_part", default="0,1", type=str)


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
        model_class = BertForPairCls   # BertForPairCls
    elif args.model_class == "roberta":
        config_class = RobertaConfig
        tokenizer_class = RobertaTokenizer
        model_class = RobertaForPairCls
    else:
        raise KeyError(args.model_class)

    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path, num_labels=2)
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

    train_dataset = LinkPredictionPairDataset(
        args.dataset, "train", None, "./data/",
        args.model_class, tokenizer, args.do_lower_case,
        args.max_seq_length, neg_times=5, neg_weights=neg_weights
    )

    dev_dataset = LinkPredictionPairDataset(
        args.dataset, "dev", None, "./data/",
        args.model_class, tokenizer, args.do_lower_case,
        args.max_seq_length,
    )



    if args.do_train:
        train(args, train_dataset, model, tokenizer, eval_dataset=dev_dataset)


    if not args.do_train and args.do_eval:
        test_dataset = LinkPredictionPairDataset(
            args.dataset, "test", None, "./data/",
            args.model_class, tokenizer, args.do_lower_case,
            args.max_seq_length,
        )
        dataset_list = [train_dataset, dev_dataset, test_dataset]
        path_template = join(args.output_dir, "tuple_ranks_{},{}.json")
        part_param = args.prediction_part.split(",")
        part_param = [int(_e) for _e in part_param]
        assert len(part_param) == 2 and part_param[1] > part_param[0] >= 0
        cur_part_idx, num_parts = part_param

        dev_raw_examples = dev_dataset.raw_examples
        tgt_raw_examples = [_ex for _idx, _ex in enumerate(dev_raw_examples) if _idx % num_parts == cur_part_idx]
        get_dev_candidate(args, tgt_raw_examples, dataset_list, model)
        print("save finished")

    if args.fp16:
        model = setup_eval_model_for_fp16(args, model)







if __name__ == '__main__':
    main()