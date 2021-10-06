import torch.nn as nn
from tqdm import tqdm, trange
from peach.help import *
import json
import collections
import matplotlib.pyplot as plt
from os.path import join
import xlwt

def rescale(arr, a=0., b=1.):
    min_val, max_val = np.min(arr), np.max(arr)
    return (arr - min_val) * ((b-a) / (max_val - min_val)) + a

def save_model(model, optimizer, args, save_variable_list):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    torch.save({
         **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.output_dir, 'checkpoint_'+args.mode+'_'+args.feature_method+'_'+str(args.train_batch_size)+'_' + str(args.learning_rate) +
                                 '_' + str(args.hinge_loss_margin) +'_'+str(args.num_train_epochs)+'_'+str(args.neg_times))
    )

def safe_ranking(_scores, _pos_idx):
    pos_score = _scores[_pos_idx]  # []
    same_score_loc = np.where(_scores == pos_score)[0]
    assert same_score_loc.size > 0
    rdm_pos_loc = same_score_loc[random.randint(0, same_score_loc.shape[0]-1)]
    _sort_idxs = np.argsort(-_scores)
    _rank = np.where(_sort_idxs == rdm_pos_loc)[0][0] + 1

    return _rank

class EnsembleModel(nn.Module):
    def __init__(self, stelp_ent_emb_path, top_k, neg_times, hinge_loss_margin):
        super(EnsembleModel, self).__init__()
        self.stelp_ent_emb = torch.load(os.path.join(stelp_ent_emb_path, "ent_emb.pkl")).cuda()
        self.emb_dim = len(self.stelp_ent_emb[0])
        self.top_k = top_k
        self.proj = nn.Linear(5 * top_k + self.emb_dim, 1)
        self.neg_times = neg_times
        self.hinge_loss_margin = hinge_loss_margin
        # self.simi_index_mtx = torch.tensor(np.load(os.path.join(stelp_ent_emb_path, "similarity_index_mtx.npy"))).cuda()
        self.simi_score_mtx = torch.tensor(np.load(os.path.join(stelp_ent_emb_path, "similarity_score_mtx.npy"))).cuda()

    def get_alpha(self, ent_idx, top_stelp_score, top_rotate_score):
        bs = ent_idx.shape[0]
        emb_mtx, simi_mtx = [], []
        for _i in range(bs):
            emb_mtx.append(self.stelp_ent_emb[ent_idx[_i]])
            simi_mtx.append(self.simi_score_mtx[ent_idx[_i]])
        simi_mtx = torch.stack(simi_mtx, dim=0)  # bs, top_k, top-M
        simi_mtx = torch.mean(simi_mtx, dim=-1)  # bs, top_k
        emb_mtx = torch.stack(emb_mtx, dim=0)  # bs, top_k, emb_dim
        emb_mtx = torch.std(emb_mtx, dim=1)  # bs, emb_dim
        score_sub = abs(top_rotate_score - top_stelp_score)
        score_add = top_stelp_score + top_rotate_score
        feature = torch.cat([emb_mtx, simi_mtx, score_sub, score_add, top_stelp_score, top_rotate_score], dim=1)
        # alpha = torch.sigmoid(self.proj_m1(feature))
        alpha = torch.sigmoid(self.proj(feature))
        return alpha


    def forward(self, pos_stelp_score, pos_rotate_score, ent_idx, neg_stelp_scores, neg_rotate_scores,
                stelp_scores, rotate_scores, method=None):
        # pos_stelp_score: bs
        # pos_rotate_score: bs
        # ent_idx : bs, top_k
        # neg_stelp_scores: bs, nt
        # neg_rotate_scores: bs, nt
        alpha = self.get_alpha(ent_idx, stelp_scores, rotate_scores).squeeze(-1)

        bs = len(alpha)

        pos_ens_score = alpha * pos_stelp_score + (1 - alpha) * pos_rotate_score  # bs
        neg_ens_score = alpha.unsqueeze(1).expand(-1, self.neg_times) * neg_stelp_scores \
                        + (1 - alpha).unsqueeze(1).expand(-1, self.neg_times) * neg_rotate_scores # bs, nt
        loss = torch.relu(self.hinge_loss_margin -
                          pos_ens_score.unsqueeze(1).expand(-1, self.neg_times).reshape(bs * self.neg_times) + neg_ens_score.view(-1))
        loss = torch.mean(loss)
        output = [loss]
        return output

    @staticmethod
    def train_step(model, optimizer, train_dataset, args, eval_dataset=None, eval_fn=None, kb_dataset_list=None,
                   g_subj2objs=None, g_obj2subjs=None):
        train_sampler = SequentialSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=args.num_workers)

        # set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
        metric_best = -1e5
        global_step = 0

        ma_dict = MovingAverageDict()
        model.zero_grad()
        train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
        for _idx_epoch, _ in enumerate(train_iterator):

            step_loss = 0
            for step, batch in enumerate(train_dataloader):
                model.train()
                batch = tuple(t.to(args.device) for t in batch)
                inputs = train_dataset.batch2feed_dict(batch)
                output = model(**inputs, method=args.feature_method)
                loss = output[0]
                step_loss += loss.item()
                loss.backward()

                if args.max_grad_norm > 0.001:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                # scheduler.step()  # Update learning rate schedule
                model.zero_grad()

                global_step += 1

                # update loss for logging
                ma_dict({"loss": step_loss})
                step_loss = 0.

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging.info(ma_dict.get_val_str())

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_variable_list = {
                        'step': global_step,
                        'current_learning_rate': args.learning_rate
                    }
                    save_model(model, optimizer, args, save_variable_list)

                if eval_dataset is not None and args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    metric_cur = eval_fn(
                        model, eval_dataset, kb_dataset_list, args, data_type="dev",
                        g_subj2objs=g_subj2objs, g_obj2subjs=g_obj2subjs)[0]
                    if metric_cur > metric_best:
                        save_variable_list = {
                            'step': global_step,
                            'current_learning_rate': args.learning_rate
                        }
                        save_model(model, optimizer, args, save_variable_list)
                        metric_best = metric_cur

            # can add epoch
        save_variable_list = {
            'step': global_step,
            'current_learning_rate': args.learning_rate
        }
        save_model(model, optimizer, args, save_variable_list)
        train_iterator.close()


    @staticmethod
    def test_step(model, test_dataset, dataset_list, args, data_type=None, g_subj2objs=None, g_obj2subjs=None):

        logger.info("--------------- Do prediction ----------------")
        model.eval()

        stelp_pos_loc, stelp_score, rotate_score, ent_idx = [], [], [], []
        context_scores = torch.load(os.path.join(args.context_score_path, data_type + "_" + args.mode + "_full_scores.list"))
        translation_scores = torch.load(os.path.join(args.translation_score_path, data_type + "_" + args.mode + "_full_scores.list"))

        for idx in range(len(context_scores)):
            stelp_pos_loc.append(int(context_scores[idx][0]))
            stelp_score.append(torch.tensor(context_scores[idx][1]))
            rotate_score.append(torch.tensor(rescale(translation_scores[idx][1])))
            ent_idx.append(torch.tensor(np.argsort(context_scores[idx][1])[:1000], dtype=torch.long))

        stelp_score = torch.stack(stelp_score, dim=0)
        rotate_score = torch.stack(rotate_score, dim=0)
        ent_idx = torch.stack(ent_idx, dim=0)

        ranks = []
        hits = []
        top_ten_hit_count = 0
        for i in range(100):
            hits.append([])

        alpha = []

        # mix
        top_stelp_score, top_rotate_score = [], []
        for i in range(len(stelp_score)):
            top_idx = np.argsort(stelp_score[i].numpy())[:1000]
            top_stelp_score.append(stelp_score[i][top_idx])
            top_rotate_score.append(rotate_score[i][top_idx])
        top_stelp_score = torch.stack(top_stelp_score, dim=0)
        top_rotate_score = torch.stack(top_rotate_score, dim=0)

        for _start in range(0, len(ent_idx), args.test_batch_size):
            if _start+args.test_batch_size < len(ent_idx):
                alpha.append(model.get_alpha(ent_idx[_start:_start+args.test_batch_size],
                                             top_stelp_score[_start:_start+args.test_batch_size].cuda(),
                                             top_rotate_score[_start:_start+args.test_batch_size].cuda()))
            else:
                alpha.append(model.get_alpha(ent_idx[_start:], top_stelp_score[_start:].cuda(), top_rotate_score[_start:].cuda()))
        alpha = torch.cat(alpha, dim=0).squeeze(1).detach().cpu()

        if args.seen_feature:
            for i in range(len(test_dataset.raw_examples)):
                if i in test_dataset.unseen_triple_id[args.mode]:
                    alpha[i] = 1


        # ---get alpha distribution figure---
        # group = [0.05 * i for i in range(0, 21)]
        # plt.cla()
        # plt.hist(alpha.numpy().tolist(), group, rwidth=0.8)
        # plt.xlabel("alpha")
        # plt.ylabel("frequent")
        # plt.title("alpha-frequent figure")
        # plt.savefig(os.path.join(args.output_dir, args.mode+'_'+args.feature_method+'_'+str(args.hinge_loss_margin)+'_' + str(args.learning_rate) +
        #                          '_' + str(args.train_batch_size) +'_'+str(args.num_train_epochs) +'_alpha_figure.jpg'))
        # print("save figure finished!")

        if args.feature_method == 'mix':
            ens_score = alpha.unsqueeze(1).expand(-1, stelp_score.shape[1]) * stelp_score \
                        + (1 - alpha).unsqueeze(1).expand(-1, rotate_score.shape[1]) * rotate_score
        elif args.feature_method == 'add':
            ens_score = stelp_score + rotate_score
        if args.get_adp_improve:
            stelp_dict = torch.load(join(args.context_score_path, 'cases_alone.dict'))
            rotate_dict = torch.load(join(args.translation_score_path, 'RotatE_case_alone.dict'))

        for idx in range(len(context_scores)):
            # step 1 : get the top_k index of context scores

            # remove other real positive sample for context_score
            _h, _r, _t = test_dataset.raw_examples[idx]
            head_remove = g_obj2subjs[_t][_r] - {_h}
            for _false_pos_head in list(head_remove):
                ens_score[idx][test_dataset.ent2idx[_false_pos_head]] = 0
            tail_remove = g_subj2objs[_h][_r] - {_t}
            for _false_pos_tail in list(tail_remove):
                ens_score[idx][test_dataset.ent2idx[_false_pos_tail]] = 0
            ex_ens_score = ens_score[idx].numpy()
            _stelp_rank = safe_ranking(stelp_score[idx].numpy(), stelp_pos_loc[idx])

            #  use the stelp_rank as the final rank if the ensemble rank not in top_k
            _rank = safe_ranking(ex_ens_score, stelp_pos_loc[idx])
            if _rank > 1000:
                _rank = _stelp_rank

            if args.get_adp_improve:
                _h = test_dataset.ent2text[_h].split(",")[0]
                _t = test_dataset.ent2text[_t].split(",")[0]
                stelp_rank = stelp_dict[(_h,_r,_t)]['tail'][0][1]
                _r = _r.replace('_', ' ')[1:]
                rotate_rank = rotate_dict[(_h, _r, _t)]['tail'][0][1]
                if _rank < stelp_rank and _rank < rotate_rank and _rank <= 10:
                    with open(join(args.context_score_path, 'adp_improve_tail_case.txt'), 'a', encoding='utf-8') as f:
                        f.write(str(str((_h,_r,_t))) + ': ')
                        f.write(str(str([_rank, stelp_rank, rotate_rank])) + '\n')


            ranks.append(_rank)
            top_ten_hit_count += int(_rank <= 10)

            # hits
            for hits_level in range(100):
                if _rank <= hits_level + 1:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)

        for i in [0, 2, 9]:
            logger.info('Hits @{0}: {1}'.format(i + 1, np.mean(hits[i])))
        logger.info('Mean rank: {0}'.format(np.mean(ranks)))
        logger.info('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))

        # if data_type == 'test':
        #     with open(os.path.join(args.output_dir, '{0}_{1}_ranks.json'), "w", encoding="utf-8") as fp:
        #         json.dump(ranks, fp)
        with open(os.path.join(args.output_dir, args.mode+'_metric.txt'), 'a') as f:
            f.write("---------{0}, {1}, lr={2}, ep={3}, nt={4}, margin={5}, bs={6} feature={7} metric ----------\n"
                    .format(args.mode, data_type, args.learning_rate, args.num_train_epochs,
                            args.neg_times, args.hinge_loss_margin, args.train_batch_size, args.feature_method))
            for i in [0, 2, 9]:
                f.write(args.mode +' Hits @{0}: {1}\n'.format(i + 1, np.mean(hits[i])))
            f.write(args.mode +'Mean rank: {0}\n'.format(np.mean(ranks)))
            f.write(args.mode +'Mean reciprocal rank: {0}\n\n'.format(np.mean(1. / np.array(ranks))))

        # save all hits infomation
        # f = xlwt.Workbook()
        # sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # create sheet
        # for i, hit in enumerate(hits):
        #     sheet1.write(i, 0, np.mean(hit))
        # f.save(join(args.output_dir, 'Adp_hits_info.csv'))  # save

        return np.mean(hits[2]), ranks

