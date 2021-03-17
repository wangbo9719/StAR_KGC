import multiprocessing
from functools import reduce


def split_to_lists(org_list, num_parallels):
    _lists = [[] for _ in range(num_parallels)]
    for _idx, _item in enumerate(org_list):
        _lists[_idx%num_parallels].append(_item)
    return _lists


def combine_from_lists(_lists, elem_type="union", ordered=False): # [list|set]
    def _aggregator(_a, _b):
        if isinstance(_a, list) and isinstance(_b, list):
            return _a + _b
        if isinstance(_a, set) and isinstance(_b, set):
            new_set = set()
            new_set.update(_a)
            new_set.update(_b)
            return new_set

    if elem_type in ["union"]:  # take into account the order when is a list
        if not ordered:
            return reduce(_aggregator, _lists)
        else:
            assert isinstance(_lists[0], list)

            num_src = len(_lists)
            max_len = max(len(_l) for _l in _lists)

            resulting_list = []
            for _idx_l in range(max_len):
                for _idx_s in range(num_src):
                    if _idx_l < len(_lists[_idx_s]):
                        resulting_list.append(_lists[_idx_s][_idx_l])
            return resulting_list

    elif elem_type in ["dict"]:
        if ordered:
            raise NotImplementedError

        resulting_dict = None
        for _dict in _lists:
            if resulting_dict is None:
                resulting_dict = _dict
            else:
                for _k, _v in _dict.items():
                    if _k in resulting_dict:
                        resulting_dict[_k] = _aggregator(resulting_dict[_k], _v)
                    else:
                        resulting_dict[_k] = _v
        return resulting_dict

def multiprocessing_map(
        func, dict_args_list,
        num_parallels,
):
    """
    Note that: all the data is async and all in parallels
    :param num_parallels:
    :return:
    """

    def _parallel_fn(_manager_dict, _process_idx, dict_args):
        _res = func(**dict_args)
        _manager_dict[_process_idx] = _res

    mng = multiprocessing.Manager()
    mng_dict = mng.dict()
    processes = []
    for _idx in range(num_parallels):
        _proc = multiprocessing.Process(
            target=_parallel_fn,
            args=(mng_dict, _idx, dict_args_list[_idx]),
        )
        processes.append(_proc)
        _proc.start()

    for _proc in processes:
        _proc.join()

    sorted_list = list(v for k, v in sorted(mng_dict.items(), key=lambda elem: elem[0]))
    assert len(sorted_list) == num_parallels, "num_parallels is {}, but got {}".format(num_parallels, len(sorted_list))
    return sorted_list






















