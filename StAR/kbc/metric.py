import logging
import os
import numpy as np
import random


def safe_ranking(_scores, verbose=False):
    pos_score = _scores[0]  # []
    same_score_loc = np.where(_scores == pos_score)[0]
    assert same_score_loc.size > 0 and same_score_loc[0] == 0
    rdm_pos_loc = same_score_loc[random.randint(0, same_score_loc.shape[0]-1)]
    _sort_idxs = np.argsort(-_scores)
    _rank = np.where(_sort_idxs == rdm_pos_loc)[0][0] + 1

    if verbose:
        _default_rank = np.where(_sort_idxs == 0)[0][0] + 1
        print("From safe_ranking: default rank is {}, after safe {}".format(_default_rank, _rank))
    return _rank

def concrete_safe_ranking(_scores, verbose=False):
    pass


def calculate_metrics_for_link_prediction(tuple_ranks, verbose=True):
    ranks_left, ranks_right, ranks = [], [], []
    hits_left, hits_right, hits = [], [], []
    for i in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])

    for _left_rank, _right_rank in tuple_ranks:
        ranks.append(_left_rank)
        ranks.append(_right_rank)
        ranks_left.append(_left_rank)
        ranks_right.append(_right_rank)

        # hits
        for hits_level in range(10):
            if _left_rank <= hits_level+1:
                hits[hits_level].append(1.0)
                hits_left[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)
                hits_left[hits_level].append(0.0)

            if _right_rank <= hits_level+1:
                hits[hits_level].append(1.0)
                hits_right[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)
                hits_right[hits_level].append(0.0)

    output_str = ""
    linesep = os.linesep

    for i in [0, 2, 9]:
        output_str += 'Hits left @{0}: {1}'.format(i + 1, np.mean(hits_left[i])) + linesep
        output_str += 'Hits right @{0}: {1}'.format(i + 1, np.mean(hits_right[i])) + linesep
        output_str += '###Hits @{0}: {1}'.format(i + 1, np.mean(hits[i])) + linesep
    output_str += 'Mean rank left: {0}'.format(np.mean(ranks_left)) + linesep
    output_str += 'Mean rank right: {0}'.format(np.mean(ranks_right)) + linesep
    output_str += '###Mean rank: {0}'.format(np.mean(ranks)) + linesep
    output_str += 'Mean reciprocal rank left: {0}'.format(np.mean(1. / np.array(ranks_left))) + linesep
    output_str += 'Mean reciprocal rank right: {0}'.format(np.mean(1. / np.array(ranks_right))) + linesep
    output_str += '###Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))) + linesep

    if verbose:
        logging.info(output_str)
    return output_str

