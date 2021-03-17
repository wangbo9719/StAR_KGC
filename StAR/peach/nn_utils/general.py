import math
import torch
import torch.nn as nn


def transform_edges_to_mask(graph_edges, seq_len, symmetry=True):
    bs = graph_edges.size(0)
    sl = seq_len
    # change graph edge [bs,n,2] to 2d index [N, 3]
    batch_idxs = torch.arange(bs).to(
        graph_edges.device).unsqueeze(-1).unsqueeze(-1).expand(bs, graph_edges.size(1), 1)  # bs,n,1
    new_graph_edges = torch.where(
        graph_edges > -1,
        graph_edges,
        graph_edges.new_full(graph_edges.size(), fill_value=sl)
    )
    g_indices = torch.cat((batch_idxs, new_graph_edges), dim=-1).contiguous().view(-1, 3)  # bs*n, 3
    # for _row in range(g_indices.size(0)):
    #     print(g_indices[_row])
    graph_mask = torch.sparse.FloatTensor(  # bs,sl,sl
        g_indices.t(), graph_edges.new_ones([g_indices.size(0)], dtype=graph_edges.dtype),
        torch.Size([bs, sl + 1, sl + 1])
    ).to_dense()[:, :-1, :-1].contiguous()

    if symmetry:
        graph_mask = (graph_mask + torch.transpose(graph_mask, -1, -2)).gt(0).to(graph_mask.dtype)

    return graph_mask  # mask with same type as `graph_edges`


def exp_mask(_mask, _val, high_rank=False):
    _exp_mask = (torch.ones_like(_mask) - _mask).to(_val.dtype) * \
                torch.full([1], fill_value=-10000, dtype=_val.dtype, device=_val.device)
    if high_rank:
        _exp_mask = _exp_mask.unsqueeze(-1).expand_as(_val)
    return _exp_mask + _val
    # the val of mask position is setted to (-10000+_val)
    # so that after softmax function(exp), the mask position value can be seemed as 0


def zero_mask(_mask, _val, high_rank=False):
    _zero_mask = _mask.to(_val.dtype)
    if high_rank:
        _zero_mask = _zero_mask.unsqueeze(-1).expand_as(_val)
    return _zero_mask * _val


def mask_2d_to_1d(mask_2d, dim=-1, threshold=0):
    return mask_2d.sum(dim).gt(threshold).to(mask_2d.dtype)


def mask_1d_to_2d(mask_1d, other_mask_1d=None):
    if other_mask_1d is None:
        other_mask_1d = mask_1d
    return mask_1d.unsqueeze(-1) * other_mask_1d.unsqueeze(-2)


def prime_to_attn_2d_mask(prime_mask):
    """

    :param prime_mask: [bs,sl] when val>0 is a prime number and val==0 is unmask
    :return:
    """
    bs = prime_mask.size(0)
    sl = prime_mask.size(1)
    mask1d = torch.where(prime_mask > 0, prime_mask, torch.ones_like(prime_mask))
    mask2d_1 = mask1d.unsqueeze(-1).expand(bs, sl, sl)
    mask2d_2 = mask1d.unsqueeze(-2).expand(bs, sl, sl)
    mask2d = (torch.remainder(mask2d_1, mask2d_2) == 0).to(prime_mask.dtype)
    # mask invalid token
    valid = (prime_mask > 0).to(prime_mask.dtype)
    valid2d_1 = valid.unsqueeze(-1).expand(bs, sl, sl)
    valid2d_2 = valid.unsqueeze(-2).expand(bs, sl, sl)

    final_mask2d = (valid2d_1*valid2d_2) * mask2d
    return final_mask2d


def masked_pool(rep_input, rep_mask, high_rank=True, method="mean", return_new_mask=False):

    dim_pool = rep_mask.dim() - 1
    new_mask = (rep_mask.sum(dim=dim_pool) > 0).to(rep_mask.dtype)

    if method == "mean":
        masked_input = zero_mask(rep_mask, rep_input, high_rank=high_rank)
        rep_output = masked_input.sum(dim=dim_pool)
        denominator = rep_mask.to(rep_output.dtype).sum(dim=dim_pool)
        # remove zero
        denominator = torch.where(
            denominator > 0.,
            denominator, torch.full_like(denominator, fill_value=1.)
        )
        if high_rank:
            denominator = denominator.unsqueeze(-1).expand_as(rep_output)
        rep_output /= denominator

    elif method == "max":
        masked_input = exp_mask(rep_mask, rep_input, high_rank=high_rank)
        rep_output = torch.max(masked_input, dim=dim_pool)[0]
    else:
        raise NotImplementedError

    rep_output = zero_mask(new_mask, rep_output, high_rank=high_rank)

    if return_new_mask:
        return rep_output, new_mask
    else:
        return rep_output


def len_mask(_lens, max_len=None):
    max_len = max_len or _lens.max().item()  # []
    rg = torch.arange(0, max_len, dtype=torch.long, device=_lens.device)  # ml
    # expand to [...] + [ml]
    for _ in range(_lens.dim()):
        rg = rg.unsqueeze(0)
    rg = rg.expand(list(_lens.size()) + [max_len])
    expd_lens = _lens.unsqueeze(-1).expand_as(rg)
    return (rg < expd_lens).to(torch.long), max_len


def slice_tensor(rep_input, rep_se):
    """

    :param rep_input: [bs,sl,hn]
    :param rep_se: [bs,nl,2]
    :return:
    """
    bs, sl, hn = rep_input.shape
    _, nl = rep_se.shape[:2]
    device = rep_input.device

    node_lens = rep_se[..., 1] - rep_se[..., 0] + 1  # bs,nl
    node_len_mask, max_node_len = len_mask(node_lens)  # [bs,nl,pl], []
    # refine node_len_mask
    node_len_mask = node_len_mask * (rep_se[..., 1] >= 0).to(torch.long).unsqueeze(-1).expand_as(node_len_mask)

    node_ranges = torch.arange(0, max_node_len, dtype=torch.long, device=device).unsqueeze(
        0).unsqueeze(0).expand([bs, nl, max_node_len])  # bs,nl,pl
    node_indices = (node_ranges + rep_se[..., 0].unsqueeze(-1).expand_as(node_ranges)) * node_len_mask
    #    - (1-node_len_mask)  # bs,nl,pl
    node_indices = node_indices.contiguous()
    node_indices_rsp = \
        node_indices.view(bs, nl*max_node_len).unsqueeze(-1).expand(bs, nl*max_node_len, hn) # bs, nl*pl, hn

    rep_node = torch.gather(rep_input, dim=1, index=node_indices_rsp).view(bs, nl, max_node_len, hn)
    rep_node = zero_mask(node_len_mask, rep_node, high_rank=True)

    return rep_node, node_len_mask  # [bs,nl,pl,hn] & [bs,nl,pl]


def reversed_slice_tensor(rep_input, rep_se, seq_len):
    """

    :param rep_input:  [bs,nl,hn]
    :param rep_se:  [bs,nl,2]
    :param seq_len: python int
    :return:
    """
    bs, nl, hn = rep_input.shape
    sl = seq_len
    device = rep_input.device
    # node to token matrix
    rgs = torch.arange(sl, device=device).unsqueeze(0).unsqueeze(0).expand([bs, nl, sl])
    start_indices = rep_se[..., 0].unsqueeze(-1).expand_as(rgs)
    end_indices = rep_se[..., 1].unsqueeze(-1).expand_as(rgs)
    node2tk_mask = (start_indices <= rgs).to(torch.long) * (rgs <= end_indices).to(torch.long)  # [bs, nl, sl]
    tk2node_mask = node2tk_mask.transpose(-1, -2)  # [bs,sl,nl]
    nums_nonzero = tk2node_mask.sum(-1)  # bs,sl
    max_nz = nums_nonzero.max().item()  # max_num_nonzero
    res_mask, _ = len_mask(nums_nonzero, max_nz)  # bs,sl,max_nz
    nums_pad = - nums_nonzero + max_nz  # bs,sl
    pad_mask, max_pd = len_mask(nums_pad)  # bs,sl,max_pd || max_padding
    pad_tk2node_mask = torch.cat([tk2node_mask, pad_mask], dim=-1)  # [bs,sl,nl+max_pd]
    res_indices = pad_tk2node_mask.nonzero()[...,-1].contiguous().view(bs, sl, max_nz)  # [bs*sl*max_nz] -> [bs,sl,max_nz]
    res_indices = zero_mask(res_mask, res_indices)
    # gather
    res_indices_gather = res_indices.view(bs, sl*max_nz).unsqueeze(-1).expand([bs, sl*max_nz, hn])
    res_gather = torch.gather(rep_input, dim=1, index=res_indices_gather).view(bs, sl, max_nz, hn)
    res_gather = zero_mask(res_mask, res_gather, high_rank=True)
    return res_gather, res_mask


# act
def act_name2fn(act_name="linear"):
    if act_name == "linear":
        return lambda x: x
    elif act_name == "relu":
        return torch.relu
    elif act_name == "gelu":
        return gelu
    else:
        KeyError(act_name)


def gelu(x):
    return (
        0.5
        * x
        * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    )