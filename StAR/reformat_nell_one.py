import argparse
import os
from tqdm import tqdm
from peach.common import save_json, save_pickle, load_json, load_pickle, load_tsv, file_exists, save_list_to_file
from wordsegment import load, segment
load()

def task_set_to_triples(data_dict):
    triple_list = []
    for vals in data_dict.values():
        triple_list.extend(vals)
    return triple_list

def reformat_nell_one():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    os.makedirs(output_dir)

    path_graph = load_tsv(os.path.join(data_dir, "path_graph"))
    rel2candidates = load_json(os.path.join(data_dir, "rel2candidates.json"))

    train_tasks = task_set_to_triples(load_json(os.path.join(data_dir, "train_tasks.json")))
    dev_tasks = task_set_to_triples(load_json(os.path.join(data_dir, "dev_tasks.json")))
    test_tasks = task_set_to_triples(load_json(os.path.join(data_dir, "test_tasks.json")))

    ent_set, rel_set = set(), set()
    for tlist in [path_graph, train_tasks, dev_tasks, test_tasks]:
        for tp in tlist:
            ent_set.add(tp[0])
            ent_set.add(tp[2])
            rel_set.add(tp[1])

    ent_list = list(sorted(ent_set))
    rel_list = list(sorted(rel_set))
    # todo save
    save_list_to_file(ent_list, os.path.join(output_dir, "entities.txt"))
    save_list_to_file(rel_list, os.path.join(output_dir, "relations.txt"))

    # ent2text rel2text
    ent2text, rel2text = [], []  # \tab split
    for ent in tqdm(ent_list):
        s = ent.split(":")
        if len(s) != 3:
            ent2text.append("{}\t{}".format(ent, ent))
        else:
            _type, _name = s[1], s[2]
            _type_tks = segment(_type)
            _name = _name.replace("__", "\'s_")
            _name_tks = _name.split("_")
            _tks = _type_tks + [":", ] + _name_tks
            ent2text.append("{}\t{}".format(ent, " ".join(_tks)))

    for rel in tqdm(rel_list):
        s = rel.split(":")
        assert len(s) == 2
        rel2text.append("{}\t{}".format(rel, " ".join(segment(s[1]))))

    save_list_to_file(ent2text, os.path.join(output_dir, "entity2text.txt"))
    save_list_to_file(rel2text, os.path.join(output_dir, "relation2text.txt"))

    # typecons.json !!!
    from collections import defaultdict
    rel2candidates = load_json(os.path.join(data_dir, "rel2candidates.json"))

    path_graph_tasks = defaultdict(list)
    for p in path_graph:
        path_graph_tasks[p[1]].append(p)

    entity_dict = defaultdict(list)
    for ent in ent_list:
        s = ent.split(':')
        if len(s) != 3:
            entity_dict['num'].append(ent)
        else:
            entity_dict[s[1]].append(ent)

    rel2candidates_in_train = defaultdict(list)
    for rel, task in path_graph_tasks.items():
        types = []
        cands = []
        for i in task:
            e1, r, e2 = i
            s = e2.split(':')
            if len(s) != 3:
                types.append('num')
            else:
                types.append(s[1])
        types = set(types)
        for t in types:
            cands.extend(entity_dict[t])
        cands = list(set(cands))
        rel2candidates_in_train[rel] = cands

    rel2candidates_new = {**rel2candidates, **rel2candidates_in_train}  # this is only head

    typecons_dict = dict()
    for rel in rel2candidates_new:
        typecons_dict[rel] = dict()
        typecons_dict[rel]["head"] = []
        typecons_dict[rel]["tail"] = rel2candidates_new[rel]
    save_json(typecons_dict, os.path.join(output_dir, "typecons.json"))

    # train dev test .tsv
    def triple_list_to_str_list(triples):
        slist = []
        for tp in triples:
            slist.append("\t".join(tp))
        return slist

    save_list_to_file(triple_list_to_str_list(path_graph + train_tasks), os.path.join(output_dir, "train.tsv"))
    save_list_to_file(triple_list_to_str_list(dev_tasks), os.path.join(output_dir, "dev.tsv"))
    save_list_to_file(triple_list_to_str_list(test_tasks), os.path.join(output_dir, "test.tsv"))


if __name__ == '__main__':
    reformat_nell_one()









