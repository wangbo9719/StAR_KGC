from peach.help import *
from torch.utils.data import Dataset, TensorDataset, DataLoader
from os.path import join
from peach.common import save_json, load_json, save_list_to_file, load_list_from_file, file_exists, \
    load_tsv
from tqdm import tqdm, trange
import collections
import logging as logger
import numpy as np
import random
import json


class KbDataset(Dataset):
    DATA_TYPE_LIST = ["train", "dev", "test","train_1900","train_918","test_alone_triples_1900"]
    NUM_REL_DICT = {
        "WN18RR": 11,
        "CN": 34,
        "CN_NEG": 34,
    }

    @staticmethod
    def build_graph(raw_examples):
        # build positive graph from triplets
        subj2objs = collections.defaultdict(lambda: collections.defaultdict(set))
        obj2subjs = collections.defaultdict(lambda: collections.defaultdict(set))

        for _raw_ex in raw_examples:
            _head, _rel, _tail = _raw_ex[:3]
            subj2objs[_head][_rel].add(_tail)
            obj2subjs[_tail][_rel].add(_head)

        return subj2objs, obj2subjs

    @staticmethod
    def build_type_constrain_dict(raw_examples):  # [type]["head"/"tail"][xxx]
        type_constrain_dict = collections.defaultdict(lambda: {"head": [], "tail": []})
        for _raw_ex in raw_examples:
            _head, _rel, _tail = _raw_ex[:3]
            type_constrain_dict[_rel]["head"].append(_head)
            type_constrain_dict[_rel]["tail"].append(_tail)
        return type_constrain_dict

    def update_negative_sampling_graph(self, raw_examples):
        # graph
        for _raw_ex in raw_examples:
            _head, _rel, _tail = _raw_ex[:3]
            self.subj2objs[_head][_rel].add(_tail)
            self.obj2subjs[_tail][_rel].add(_head)

        if self.pos_triplet_str_set is not None:
            self.pos_triplet_str_set.update(set(self._triplet2str(_ex) for _ex in raw_examples))

    def pre_negative_sampling(self, pos_raw_examples, neg_times):
        neg_raw_example_lists = []
        for _pos_raw_ex in pos_raw_examples:
            neg_raw_example_list = []
            for _ in range(neg_times):
                neg_raw_example_list.append(self.negative_sampling(_pos_raw_ex, self.neg_weights))
            neg_raw_example_lists.append(neg_raw_example_list)
        return neg_raw_example_lists

    def __init__(
            self, dataset, data_type, data_format, all_dataset_dir,
            tokenizer_type, tokenizer, do_lower_case,
            max_seq_length, neg_times=0, neg_weights=None,
            type_cons_neg_sample=False, type_cons_ratio=0, *args, **kwargs
    ):
        # assert data_type in self.DATA_TYPE_LIST
        self.dataset = dataset
        self.data_type = data_type
        self.data_format = data_format
        self.all_dataset_dir = all_dataset_dir

        self.tokenizer_type = tokenizer_type
        self.tokenizer = tokenizer
        self.do_lower_case = do_lower_case
        self.max_seq_length = max_seq_length
        self.neg_times = neg_times
        self.neg_weights = neg_weights  # a list whose len=3: relative weight for head, tail, rel corruption
        self.type_cons_neg_sample = type_cons_neg_sample
        self.type_cons_ratio = type_cons_ratio

        assert self.tokenizer_type in ["bert", "roberta"]
        assert self.dataset in ["WN18RR", "FB15k-237", "UMLS", "NELL_standard"]

        for _key, _val in kwargs.items():
            setattr(self, _key, _val)

        self.data_dir = join(all_dataset_dir, self.dataset)
        self.data_path = join(self.data_dir, "{}.tsv".format(self.data_type))

        self.ent_list, self.rel_list, self.ent2text, self.rel2text = self._read_ent_rel_info()
        self.ent2idx = dict((_e, _idx) for _idx, _e in enumerate(self.ent_list))  # useless for bert base kg
        self.rel2idx = dict((_e, _idx) for _idx, _e in enumerate(self.rel_list))

        self.raw_examples = self._read_raw_examples(self.data_path)
        self.subj2objs, self.obj2subjs = None, None
        if self.data_type == "train":  # build graph to support advanced negative sampling
            self.subj2objs, self.obj2subjs = self.build_graph(self.raw_examples)
        # to support all negative sampling
        self.pos_triplet_str_set = set(self._triplet2str(_ex) for _ex in self.raw_examples)

        # special ids for text features
        self._sep_id, self._cls_id, self._pad_id = self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.sep_token, self.tokenizer.cls_token, self.tokenizer.pad_token,]
        )

        # if needed, pre-sampled negative examples
        self.negative_samples = None
        # for some special dataset which needs pre-defined text
        self.triplet2text = None
        if self.dataset == 'NELL_standard':
            with open(join(self.data_dir, "typecons.json"), "r") as f:
                self.type_dict = json.load(f)

    def _read_ent_rel_info(self):
        # read entities and relations from files
        # entity list rel list
        ent_list = load_list_from_file(join(self.data_dir, "entities.txt"))
        rel_list = load_list_from_file(join(self.data_dir, "relations.txt"))
        # read entities and relations's text from file
        ent2text = dict(tuple(_line.strip().split('\t'))
                        for _line in load_list_from_file(join(self.data_dir, "entity2text.txt")))
        if self.data_dir.find("FB15") != -1:
            print("FB15k-237 with description")
            with open(os.path.join(self.data_dir, "entity2textlong.txt"), 'r') as f:
                ent_lines = f.readlines()
                for line in ent_lines:
                    temp = line.strip().split('\t')
                    #first_sent_end_position = temp[1].find(".")
                    ent2text[temp[0]] = temp[1]#[:first_sent_end_position + 1]

        rel2text = dict(tuple(_line.strip().split('\t'))
                        for _line in load_list_from_file(join(self.data_dir, "relation2text.txt")))
        return ent_list, rel_list, ent2text, rel2text

    def _read_raw_examples(self, data_path):  # readlines from tsv file
        examples = []
        lines = load_tsv(data_path)
        for _idx, _line in enumerate(lines):
            examples.append(_line)
        return examples

    def _triplet2str(self, raw_kg_triplet):  # this str is used to confirm uniqueness of a triplet
        return "\t".join(raw_kg_triplet[:3])

    def negative_sampling(self, raw_kg_triplet, weights=None, *args, **kwargs):
        head, rel, tail = raw_kg_triplet[:3]
        if weights is None:
            weights = [1., 1., 1.]
        cdf = np.cumsum(np.array(weights) / sum(weights))

        prob = random.random()

        neg_example = [head, rel, tail]
        if self.dataset == 'NELL_standard':
            while True:
                # do for tail entity
                pos_ent_list = self.subj2objs[head][rel]
                pos_ent_set = set(pos_ent_list)

                sample_list = []
                tail_set = set()
                tail_set.add(tail)

                neg_elem = None
                max_iter = 1000
                while max_iter > 0:
                    neg_elem = random.choice(self.type_dict[rel]["tail"])
                    if neg_elem not in pos_ent_set:
                        break
                    max_iter -= 1
                if max_iter == 0:
                    neg_elem = None
                    max_iter = 1000
                    while max_iter > 0:
                        neg_elem = random.choice(self.ent_list)
                        if neg_elem not in pos_ent_set:
                            break
                        max_iter -= 1
                    if max_iter == 0:
                        print("Warning: max iter reached when negative sampling, chose from pos set")

                assert neg_elem is not None
                neg_example[2] = neg_elem

                if self.pos_triplet_str_set is not None and self._triplet2str(neg_example) in self.pos_triplet_str_set:
                    neg_example = [head, rel, tail]
                    continue
                else:
                    break
            return neg_example

        if self.data_type != "train":  # sampling for test and dev if needed
            while True:
                if prob < cdf[0]:
                    src_elem = neg_example[0]
                    while True:
                        rdm_elem = random.choice(self.ent_list)
                        if src_elem != rdm_elem:
                            break
                    assert rdm_elem is not None
                    neg_example[0] = rdm_elem

                elif prob < cdf[1]:
                    src_elem = neg_example[2]
                    while True:
                        rdm_elem = random.choice(self.ent_list)
                        if src_elem != rdm_elem:
                            break
                    assert rdm_elem is not None
                    neg_example[2] = rdm_elem
                else:
                    src_elem = neg_example[1]
                    while True:
                        rdm_elem = random.choice(self.rel_list)
                        if src_elem != rdm_elem:
                            break
                    assert rdm_elem is not None
                    neg_example[1] = rdm_elem
                if self.pos_triplet_str_set is not None and self._triplet2str(neg_example) in self.pos_triplet_str_set:
                    continue
                else:
                    break

            return neg_example

        while True:
            if prob < cdf[0]:
                # do for head entity
                pos_ent_list = self.obj2subjs[tail][rel]
                pos_ent_set = set(pos_ent_list)
                neg_elem = None
                max_iter = 1000
                while max_iter > 0:
                    neg_elem = random.choice(self.ent_list)
                    if neg_elem not in pos_ent_set:
                        break
                    max_iter -= 1
                if max_iter == 0:
                    print("Warning: max iter reached when negative sampling, chose from pos set")

                assert neg_elem is not None
                neg_example[0] = neg_elem

            elif prob < cdf[1]:
                # do for tail entity
                pos_ent_list = self.subj2objs[head][rel]
                pos_ent_set = set(pos_ent_list)
                neg_elem = None
                max_iter = 1000
                while max_iter > 0:
                    neg_elem = random.choice(self.ent_list)
                    if neg_elem not in pos_ent_set:
                        break
                    max_iter -= 1
                if max_iter == 0:
                    print("Warning: max iter reached when negative sampling, chose from pos set")

                assert neg_elem is not None
                neg_example[2] = neg_elem

            else:
                src_elem = neg_example[1]
                while True:
                    rdm_elem = random.choice(self.rel_list)
                    if src_elem != rdm_elem:
                        break
                assert rdm_elem is not None
                neg_example[1] = rdm_elem
            if self.pos_triplet_str_set is not None and self._triplet2str(neg_example) in self.pos_triplet_str_set:
                neg_example = [head, rel, tail]
                continue
            else:
                break
        return neg_example

    def str2ids(self, text, max_len=None):
        if self.do_lower_case and self.tokenizer_type == "bert":
            text = text.lower()
        text = self.tokenizer.cls_token + " " + text
        wps = self.tokenizer.tokenize(text)
        if max_len is not None:
            wps = self.tokenizer.tokenize(text)[:max_len]
        wps.append(self.tokenizer.sep_token)
        return self.tokenizer.convert_tokens_to_ids(wps)

    def convert_raw_example_to_features(self, raw_kg_triplet, method="0"):
        head, rel, tail = raw_kg_triplet[:3]

        if method == "0":
            head_ids = self.str2ids("Node: " + self.ent2text[head])
            relidx = self.rel2idx[rel]
            tail_ids = self.str2ids("Node: " + self.ent2text[tail])
            return head_ids, relidx, tail_ids
        elif method == "1":
            head_ids = self.str2ids("Node: " + self.ent2text[head])
            rel_ids = self.str2ids("Rel: " + self.rel2text[rel])
            tail_ids = self.str2ids("Node: " + self.ent2text[tail])
            return head_ids, rel_ids, tail_ids
        elif method == "2":
            # combine
            rel_ids = self.str2ids(self.rel2text[rel])[1:-1]  # [1:-1] for removing special token from self.str2ids
            remain_len = self.max_seq_length - 4 - len(rel_ids)
            assert remain_len >= 4  # need sufficient budget for entities' ids
            head_ids = self.str2ids(self.ent2text[head])[1:-1]
            tail_ids = self.str2ids(self.ent2text[tail])[1:-1]
            while len(head_ids) + len(tail_ids) > remain_len:
                if len(head_ids) > len(tail_ids):
                    head_ids.pop(-1)
                else:
                    tail_ids.pop(-1)
            input_ids = [self._cls_id] + head_ids + [self._sep_id] + rel_ids + [self._sep_id] + tail_ids +[self._sep_id]
            type_ids = [0] * (len(head_ids)+2) + [1] * (len(rel_ids)+1) + [0] * (len(tail_ids)+1)
            return input_ids, type_ids
        elif method == "3":
            assert self.triplet2text is not None
            triplet_str = "\t".join([head, rel, tail])
            gen_sent = self.triplet2text[triplet_str]
            input_ids = self.str2ids(gen_sent, self.max_seq_length)
            type_ids = [0] * len(input_ids)
            return input_ids, type_ids
        elif method == "4":
            head_ids = self.str2ids(self.ent2text[head])
            rel_ids = self.str2ids(self.rel2text[rel])
            tail_ids = self.str2ids(self.ent2text[tail])
            return head_ids, rel_ids, tail_ids
        elif method == "5":
            head_ids = self.str2ids(self.ent2text[head])[1:-1]
            rel_ids = self.str2ids(self.rel2text[rel])[1:-1]
            tail_ids = self.str2ids(self.ent2text[tail])[1:-1]
            return head_ids, rel_ids, tail_ids
        else:
            raise KeyError(method)

    def __getitem__(self, item):
        if self.data_type == "train":  # this is for negative sampling
            assert self.subj2objs is not None and self.obj2subjs is not None

        if self.neg_times > 0:
            pos_item = item // (1 + self.neg_times)
            if item % (1 + self.neg_times) == 0:
                label = 1
                raw_ex = self.raw_examples[pos_item]
            else:
                label = 0
                if self.negative_samples is None:
                    pos_raw_ex = self.raw_examples[pos_item]
                    raw_ex = self.negative_sampling(pos_raw_ex, self.neg_weights)
                else:
                    neg_exs = self.negative_samples[pos_item]
                    if isinstance(neg_exs, list) and len(neg_exs) in [3, 4] \
                            and all(isinstance(_e, (str, int)) for _e in neg_exs):
                        raw_ex = neg_exs
                    else:
                        neg_idx = item % (1 + self.neg_times) - 1  # from [1,self.neg_times] -> [0, self.neg_times-1]
                        raw_ex = neg_exs[neg_idx % len(neg_exs)]
        else:
            raw_ex = self.raw_examples[item]

            if len(raw_ex) > 3:  # 为什么会大于3？  训练集的label哪来的？
                label = int(float(raw_ex[3]) > 0)
            elif self.data_type in ["dev", "test"]:  # if no label in "dev", "test", default is "positive"
                label = 1
            else:
                raise AttributeError
            raw_ex = raw_ex[:3]

        return raw_ex, label

    def __len__(self):
        return len(self.raw_examples) * (1 + self.neg_times)

    def data_collate_fn(self, batch):
        tensors_list = list(zip(*batch))
        return_list = []
        for _idx_t, _tensors in enumerate(tensors_list):
            if _idx_t % 3 == 0:
                padding_value = self._pad_id
            else:
                padding_value = 0

            if _tensors[0].dim() >= 1:
                return_list.append(
                    torch.nn.utils.rnn.pad_sequence(_tensors, batch_first=True, padding_value=padding_value),
                )
            else:
                return_list.append(torch.stack(_tensors, dim=0))
        return tuple(return_list)


