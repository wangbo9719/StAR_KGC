import torch
import numpy as np
from torch.utils.data import DataLoader
from dataloader import TestDataset
import logging
import os
import collections
from tqdm import tqdm
def test_get_cases(model, test_triples, all_true_triples, args):
    '''
    Evaluate the model on test or valid datasets
    '''

    model.eval()

    case_dict_path = os.path.join(args.data_path, 'RotatE_case_alone.dict')
    case_text_path = os.path.join(args.data_path, 'RotatE_case_alone.txt')
    test_dataloader_head = DataLoader(
        TestDataset(
            test_triples,
            all_true_triples,
            args.nentity,
            args.nrelation,
            'head-batch'
        ),
        batch_size=args.test_batch_size,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=TestDataset.collate_fn
    )

    test_dataloader_tail = DataLoader(
        TestDataset(
            test_triples,
            all_true_triples,
            args.nentity,
            args.nrelation,
            'tail-batch'
        ),
        batch_size=args.test_batch_size,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=TestDataset.collate_fn
    )

    test_dataset_list = [test_dataloader_head, test_dataloader_tail]

    logs = []

    step = 0
    total_steps = sum([len(dataset) for dataset in test_dataset_list])

    eid2text_dict = torch.load(os.path.join(args.data_path, 'id2text.dict'))
    rid2text_dict = torch.load(os.path.join(args.data_path, 'rid2text.dict'))
    results_dict = collections.defaultdict(dict)


    with torch.no_grad():
        for mode_id, test_dataset in enumerate(test_dataset_list):
            for _, (positive_sample, negative_sample, filter_bias, mode) in enumerate(tqdm(test_dataset, desc="get cases")):
                if args.cuda:
                    positive_sample = positive_sample.cuda()
                    negative_sample = negative_sample.cuda()
                    filter_bias = filter_bias.cuda()

                batch_size = positive_sample.size(0)

                score = model((positive_sample, negative_sample), mode)
                score += filter_bias

                # Explicitly sort all the entities to ensure that there is no test exposure bias
                argsort = torch.argsort(score, dim=1, descending=True)

                if mode == 'head-batch':
                    positive_arg = positive_sample[:, 0]
                elif mode == 'tail-batch':
                    positive_arg = positive_sample[:, 2]
                else:
                    raise ValueError('mode %s not supported' % mode)

                for i in range(batch_size):
                    # Notice that argsort is not ranking
                    ranking = (argsort[i, :] == positive_arg[i]).nonzero()  # Numpy.nonzero()返回的是数组中，非零元素的位置
                    assert ranking.size(0) == 1

                    part_sort = argsort[i,:][:20]

                    # get positive triple
                    h_id, r_id, t_id = positive_sample[i]
                    h_text = eid2text_dict[int(h_id)]
                    r_text = rid2text_dict[int(r_id)]
                    t_text = eid2text_dict[int(t_id)]
                    triples_text = [h_text, r_text, t_text]

                    sort_result = []
                    sort_result.append([eid2text_dict[int(positive_arg[i])], (ranking + 1).detach().cpu().item()])
                    for rk, id in enumerate(part_sort):
                        ent = eid2text_dict[int(id)]
                        rank = rk + 1
                        sort_result.append([ent, rank])

                    if mode_id == 0:
                        results_dict[tuple(triples_text)]["head"] = sort_result
                    elif mode_id == 1:
                        results_dict[tuple(triples_text)]["tail"] = sort_result

                step += 1

    torch.save(results_dict, case_dict_path)
    with open(case_text_path, 'a', encoding='utf-8') as f:
        for triple, detail in results_dict.items():

            f.write(str(triple) + '\n')
            f.write("head:" + str(results_dict[triple]["head"]) + '\n')
            f.write("tail:" + str(results_dict[triple]["tail"]) + '\n\n')

