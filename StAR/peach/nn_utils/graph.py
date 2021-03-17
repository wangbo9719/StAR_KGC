import torch
import torch.nn as nn

from src.nn_utils.general import mask_2d_to_1d, mask_1d_to_2d, zero_mask, transform_edges_to_mask, exp_mask
from src.nn_utils.nn import BilinearAttn
from src.nn_utils.general import transform_edges_to_mask, mask_2d_to_1d, zero_mask, exp_mask, prime_to_attn_2d_mask,\
    masked_pool, slice_tensor, reversed_slice_tensor, len_mask


def get_node_rep_from_context(sequence_output, node_ses):
    bs, nctk, nl, _ = node_ses.shape
    node_ses_rsp = node_ses.contiguous().view(bs, nctk * nl, 2)  # [bs,nctk*nl,2]
    node_reps_rsp, node_reps_mask_rsp = slice_tensor(
        sequence_output, node_ses_rsp)  # [bs,nctk*nl,pl,hn]-[bs,nctk*nl,pl]
    node_rep_rsp, node_rep_mask_rsp = masked_pool(  # [bs,nctk*nl,hn]-[bs,nctk*nl]
        node_reps_rsp, node_reps_mask_rsp, high_rank=True, method="mean", return_new_mask=True)
    node_rep = node_rep_rsp.view(bs, nctk, nl, node_rep_rsp.size(-1))
    node_rep_mask = node_rep_mask_rsp.view(bs, nctk, nl)
    ctk_rep, ctk_mask = masked_pool(  # [ns,nctk,hn] - [ns,nctk]
        node_rep, node_rep_mask, high_rank=True, method="mean", return_new_mask=True)
    return ctk_rep, ctk_mask


def combine_two_sequence(node_ctx_pos_ids, node_ctx_neg_ids, new_ml, padding_idx):
    ncsl = max(node_ctx_pos_ids.shape[-1], node_ctx_neg_ids.shape[-1])
    padded_node_ctx_ids = []
    for _tensor in [node_ctx_pos_ids[:, :new_ml or 1], node_ctx_neg_ids[:, :new_ml or 1]]:
        _tensor_pad_size = list(_tensor.shape)
        _tensor_pad_size[-1] = ncsl - _tensor_pad_size[-1]
        _tenor_padded = torch.cat([_tensor, _tensor.new_full(_tensor_pad_size, padding_idx)], dim=-1)
        padded_node_ctx_ids.append(_tenor_padded)
    node_ctx_ids = torch.cat(padded_node_ctx_ids, dim=1)
    return node_ctx_ids


def gather_masked_node_reps(node_masked_lens, ml, node_masked_flag, node_masked_mask, ctk_rep):
    bs = ctk_rep.shape[0]
    node_masked_pad_lens = ml - node_masked_lens
    node_masked_pad_mask, _ = len_mask(node_masked_pad_lens)  # [bs, YY]
    node_masked_flag_padded = torch.cat([node_masked_flag, node_masked_pad_mask], dim=-1).contiguous()  # [bs,nckt+YY]
    node_masked_flag_padded_nz = node_masked_flag_padded.nonzero().view(bs, ml, 2)[..., 1]  # [bs*ml,2] -> [bs,ml]
    node_masked_flag_padded_nz = zero_mask(node_masked_mask, node_masked_flag_padded_nz)  # [bs,ml] remove OutOfIndex
    # 5) begin to gather
    node_masked_flag_padded_nz_epd = torch.unsqueeze(node_masked_flag_padded_nz, -1).expand(
        [-1, -1, ctk_rep.size(-1)])  # [bs,ml,hn]
    node_pool_rep = torch.gather(ctk_rep, dim=1, index=node_masked_flag_padded_nz_epd)  # [bs,ml,hn]
    return node_pool_rep


class GATParamV1(nn.Module):
    def __init__(self, hn):
        super(GATParamV1, self).__init__()
        self._gat_bilinear1 = BilinearAttn(hn, hn)
        self._gat_bilinear2 = BilinearAttn(hn, hn)

        self._attn_node2rel = BilinearAttn(hn, hn)
        self._attn_rel2node = BilinearAttn(hn, hn)

    def forward(self, hidden_states, attention_mask, graph_adj_mat):
        bs, sl, hn = hidden_states.size()
        # attn_pair_mask = mask_1d_to_2d(attention_mask)  # bs,sl,sl
        # change graph edge [bs,n,2] to 2d index [N, 3]
        graph_mask = graph_adj_mat
        concept_mask = mask_2d_to_1d(graph_mask)
        non_concept_mask = (1 - concept_mask) * attention_mask

        # attn
        res_hop1 = self._gat_bilinear1(hidden_states, hidden_states, hidden_states, graph_mask)
        res_gat = self._gat_bilinear2(res_hop1, res_hop1, res_hop1, graph_mask)

        # NODE 2 REL
        node2rel_mask = mask_1d_to_2d(concept_mask, non_concept_mask)
        node2rel_res = self._attn_node2rel(res_gat, res_gat, res_gat, node2rel_mask)

        # REL 2 NODE
        rel2node_mask = mask_1d_to_2d(non_concept_mask, concept_mask)
        rel2node_res = self._attn_rel2node(res_gat, res_gat, res_gat, rel2node_mask)

        # final res
        return res_gat + zero_mask(concept_mask, node2rel_res, high_rank=True) + \
               zero_mask(non_concept_mask, rel2node_res, high_rank=True)


class GATNonparam(nn.Module):
    def __init__(self):
        super(GATNonparam, self).__init__()
        self._attn_softmax = torch.nn.Softmax(dim=-1)

    def forward(self, hidden_states, rep_mask, attn_mask, **kwargs):
        bs, sl, hn = hidden_states.size()
        # co
        attn_scores = torch.bmm(  # bs,sl,sl
            hidden_states, torch.transpose(hidden_states, 1, 2))/(sl ** 0.5)
        # change graph edge [bs,n,2] to 2d index [N, 3]
        graph_mask = attn_mask

        attn_prob = self._attn_softmax(exp_mask(graph_mask, attn_scores))  # bs,sl,sl
        attn_res = torch.bmm(attn_prob, hidden_states)  # [bs,sl,sl]x[bs,sl,hn] ==> [bs,sl,hn]
        final_res = zero_mask(rep_mask, attn_res, high_rank=True)
        return final_res
