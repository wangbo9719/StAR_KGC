import torch
from torch.utils.data import Dataset
from os.path import join
import random
import collections
torch.set_printoptions(precision=8)
from peach.common import save_json, load_json, save_list_to_file, load_list_from_file, file_exists, \
    load_tsv
import os

class EnsembleDataset(Dataset):
    def __init__(self, data_type, mode, score_path, neg_times):
        self.scores_info = torch.load(join(score_path, data_type + "_ensemble_" + mode + "_dataset.list"))
        self.neg_times = neg_times
        self.top_k = len(self.scores_info[0][1])

    def __len__(self):
        return len(self.scores_info)

    def __getitem__(self, idx):
        stelp_pos_loc, stelp_score, rotate_score, ent_idx = \
            self.scores_info[idx][0],self.scores_info[idx][1],self.scores_info[idx][2],self.scores_info[idx][3]

        tolerate = 200
        neg_idx_set = set()
        neg_stelp_score = []
        neg_rotate_score = []
        while len(neg_idx_set) < self.neg_times and tolerate > 0:
            neg_raw_idx = random.randint(0, self.top_k - 1)
            if neg_raw_idx not in neg_idx_set and neg_raw_idx != stelp_pos_loc:
                neg_idx_set.add(neg_raw_idx)
                neg_stelp_score.append(torch.tensor(stelp_score[neg_raw_idx]))
                neg_rotate_score.append(torch.tensor(rotate_score[neg_raw_idx]))
            tolerate -= 1
        # assert len(neg_raw_ex_list) == 0
        if len(neg_idx_set) < self.neg_times:
            neg_stelp_score = [neg_stelp_score[idx % len(neg_stelp_score)] for idx in range(self.neg_times)]
            neg_rotate_score = [neg_rotate_score[idx % len(neg_rotate_score)] for idx in range(self.neg_times)]
        neg_stelp_score = torch.stack(neg_stelp_score, dim=0)
        neg_rotate_score = torch.stack(neg_rotate_score, dim=0)

        return (torch.tensor(stelp_score[stelp_pos_loc]),  # pos stelp score
                torch.tensor(rotate_score[stelp_pos_loc]),  # pos rotate score
                torch.tensor(ent_idx,dtype=torch.long),  # top-N ent index
                neg_stelp_score, # neg stelp scores
                neg_rotate_score, # neg rotate scores
                torch.tensor(stelp_score),
                torch.tensor(rotate_score)
                # torch.tensor(stelp_pos_loc, dtype=torch.int)  # pos loc
                )

    def data_collate_fn(self, batch):
        pos_stelp_score, pos_rotate_score, ent_idx, neg_stelp_scores, neg_rotate_scores,\
            stelp_scores, rotate_scores = zip(*batch)
        return pos_stelp_score, pos_rotate_score, ent_idx, neg_stelp_scores, neg_rotate_scores, \
               stelp_scores, rotate_scores
    #
    @classmethod
    def batch2feed_dict(cls, batch):
        inputs = {
            'pos_stelp_score': batch[0],
            'pos_rotate_score': batch[1],  #
            'ent_idx': batch[2],  #
            'neg_stelp_scores': batch[3],  #
            'neg_rotate_scores': batch[4],  #
            'stelp_scores': batch[5],  #
            'rotate_scores': batch[6],  #
        }
        return inputs

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

    def __init__(
            self, dataset, data_type, data_format, all_dataset_dir,*args, **kwargs
    ):
        # assert data_type in self.DATA_TYPE_LIST
        self.dataset = dataset
        self.data_type = data_type
        self.data_format = data_format
        self.all_dataset_dir = all_dataset_dir
        for _key, _val in kwargs.items():
            setattr(self, _key, _val)

        self.data_dir = join(all_dataset_dir, self.dataset)
        self.data_path = join(self.data_dir, "{}.tsv".format(self.data_type))

        self.ent_list, self.rel_list, self.ent2text, self.rel2text = self._read_ent_rel_info()
        self.ent2idx = dict((_e, _idx) for _idx, _e in enumerate(self.ent_list))  # useless for bert base kg
        self.rel2idx = dict((_e, _idx) for _idx, _e in enumerate(self.rel_list))
        self.unseen_triple_id = torch.load(join(self.data_dir, 'unseen_triple_id.dict'))

        self.raw_examples = self._read_raw_examples(self.data_path)
        self.subj2objs, self.obj2subjs = None, None
        if self.data_type == "train":  # build graph to support advanced negative sampling
            self.subj2objs, self.obj2subjs = self.build_graph(self.raw_examples)
            # if type_cons_neg_sample:
        #self.type_constrain_dict = self.build_type_constrain_dict(self.raw_examples)
        # to support all negative sampling
        self.pos_triplet_str_set = set(self._triplet2str(_ex) for _ex in self.raw_examples)

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

    def str2ids(self, text, max_len=None):
        if self.do_lower_case and self.tokenizer_type == "bert":
            text = text.lower()
        text = self.tokenizer.cls_token + " " + text
        wps = self.tokenizer.tokenize(text)
        if max_len is not None:
            wps = self.tokenizer.tokenize(text)[:max_len]
        wps.append(self.tokenizer.sep_token)
        return self.tokenizer.convert_tokens_to_ids(wps)


    def __getitem__(self, item):
        return None

    def __len__(self):
        return len(self.raw_examples)


