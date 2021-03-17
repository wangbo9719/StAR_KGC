from tqdm import tqdm, trange
from peach.help import *
import collections
from kbc.metric import safe_ranking


def evaluate(args, eval_dataset, model, tokenizer, global_step=None, file_prefix=""):
    def str2ids(text, max_len=None):
        if args.do_lower_case:
            text = text.lower()
        wps = tokenizer.tokenize(text)
        if max_len is not None:
            wps = tokenizer.tokenize(text)[:max_len]
        return tokenizer.convert_tokens_to_ids(wps)

    logging.info("***** Running evaluation at {}*****".format(global_step))
    logging.info("  Num examples = %d", len(eval_dataset))
    model.eval()

    # sample on dict keys to get sample_raw_examples, then use it to replace all raw_examples
    # get data

    dev_dict = torch.load("./data/"+args.dataset+"FB15k-237/new_dev.dict")
    neg_num = 50
    raw_examples = list(dev_dict.keys())
    # sample
    sample_raw_examples = [raw_examples[i] for i in range(0, len(raw_examples), 6)]
    rel_list = list(sorted(eval_dataset.rel_list))
    all_tail_ents = []
    for triplet in sample_raw_examples:
        _,_,_tail = triplet
        all_tail_ents.append(_tail)
        all_tail_ents.extend(dev_dict[triplet]["tails_corrupt"])
    all_tail_ents = list(set(all_tail_ents))  # 2612

    input_ids_list, mask_ids_list, segment_ids_list = [], [], []
    t_input_ids_list, t_mask_ids_list, t_segment_ids_list = [], [], []
    for _triplet in tqdm(sample_raw_examples):
        _head, _rel, _tail = _triplet
        all_heads = dev_dict[_triplet]["heads_corrupt"]
        all_heads.insert(0, _head)   # pos_head, + heads_corrupt
        rel_ids = str2ids(eval_dataset.rel2text[_rel])
        max_ent_len = eval_dataset.max_seq_length - 3 - len(rel_ids)
        for i, _ in enumerate(all_heads):
            head_ids = str2ids(eval_dataset.ent2text[all_heads[i]])
            head_ids = head_ids[:max_ent_len]

            # get head + rel input ids
            src_input_ids = [eval_dataset._cls_id] + head_ids + [eval_dataset._sep_id] + rel_ids + [
                eval_dataset._sep_id]
            src_mask_ids = [1] * len(src_input_ids)
            src_segment_ids = [0] * (len(head_ids) + 2) + [1] * (len(rel_ids) + 1)
            input_ids_list.append(src_input_ids)
            mask_ids_list.append(src_mask_ids)
            segment_ids_list.append(src_segment_ids)

    for tail_ent in all_tail_ents:
        max_ent_len = eval_dataset.max_seq_length - 3
        tail_ids = str2ids(eval_dataset.ent2text[tail_ent])
        tail_ids = tail_ids[:max_ent_len]
        tgt_input_ids = [eval_dataset._cls_id] + tail_ids + [eval_dataset._sep_id]
        tgt_mask_ids = [1] * len(tgt_input_ids)
        tgt_segment_ids = [0] * (len(tail_ids) + 2)
        input_ids_list.append(tgt_input_ids)
        mask_ids_list.append(tgt_mask_ids)
        segment_ids_list.append(tgt_segment_ids)

    # # padding
    max_len = max(len(_e) for _e in input_ids_list)
    assert max_len <= eval_dataset.max_seq_length
    input_ids_list = [_e + [eval_dataset._pad_id] * (max_len - len(_e)) for _e in input_ids_list]
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

    # ------------------------  get all embeddings ----------------------------
    print("\tget all emb via model")
    embs_list = []
    for batch in tqdm(enc_dataloader, desc="embedding"):
        batch = tuple(t.to(args.device) for t in batch)
        _input_ids, _mask_ids, _segment_ids = batch
        with torch.no_grad():
            embs = model.encoder(_input_ids, attention_mask=_mask_ids, token_type_ids=_segment_ids)
            embs = embs.detach().cpu()
            embs_list.append(embs)
    emb_mat = torch.cat(embs_list, dim=0).contiguous()
    assert emb_mat.shape[0] == len(input_ids_list)

    # -------------------------  assign to ent ------------------------------

    ent_rel2emb = collections.defaultdict(dict)
    ent2emb = dict()
    ptr_row = 0
    for _triplet in sample_raw_examples:
        _head, _rel, _tail = _triplet
        all_heads = dev_dict[_triplet]["heads_corrupt"]
        for i, _ in enumerate(all_heads):
            ent_rel2emb[all_heads[i]][_rel] = emb_mat[ptr_row]
            ptr_row += 1
    for _tail in all_tail_ents:
        ent2emb[_tail] = emb_mat[ptr_row]
        ptr_row += 1


    # ---------------------------- use HIT@10 as metric -----------------------------
    ranks_left, ranks_right, ranks = [], [], []
    hits_left, hits_right, hits = [], [], []
    top_ten_hit_count = 0
    for i in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])

    for _idx_ex, _triplet in enumerate(tqdm(sample_raw_examples, desc="evaluating")):
        _head, _rel, _tail = _triplet

        head_ent_list = []
        tail_ent_list = []

        # head corrupt
        _neg_head_ents = dev_dict[_triplet]["heads_corrupt"]
        head_ent_list.append(_head)  # positive example
        head_ent_list.extend(_neg_head_ents)  # negative examples  num = 50
        tail_ent_list.extend([_tail] * (1 + len(_neg_head_ents)))
        split_idx = len(head_ent_list)

        # tail corrupt
        _neg_tail_ents = dev_dict[_triplet]["tails_corrupt"]
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
        left_rank = safe_ranking(left_scores)
        ranks_left.append(left_rank)
        ranks.append(left_rank)

        # right
        right_scores = scores[split_idx:]
        right_rank = safe_ranking(right_scores)
        ranks_right.append(right_rank)
        ranks.append(right_rank)

        # log
        top_ten_hit_count += (int(left_rank <= 10) + int(right_rank <= 10))
        if (_idx_ex + 1) % 10 == 0:
            logger.info("hit@10 until now: {}".format(top_ten_hit_count * 1.0 / len(ranks)))
            logger.info('mean rank until now: {}'.format(np.mean(ranks)))

        # hits
        for hits_level in range(10):
            if left_rank <= hits_level + 1:
                hits[hits_level].append(1.0)
                hits_left[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)
                hits_left[hits_level].append(0.0)

            if right_rank <= hits_level + 1:
                hits[hits_level].append(1.0)
                hits_right[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)
                hits_right[hits_level].append(0.0)

    for i in [0, 2, 9]:
        logger.info('Hits left @{0}: {1}'.format(i + 1, np.mean(hits_left[i])))
        logger.info('Hits right @{0}: {1}'.format(i + 1, np.mean(hits_right[i])))
        logger.info('Hits @{0}: {1}'.format(i + 1, np.mean(hits[i])))
    logger.info('Mean rank left: {0}'.format(np.mean(ranks_left)))
    logger.info('Mean rank right: {0}'.format(np.mean(ranks_right)))
    logger.info('Mean rank: {0}'.format(np.mean(ranks)))
    logger.info('Mean reciprocal rank left: {0}'.format(np.mean(1. / np.array(ranks_left))))
    logger.info('Mean reciprocal rank right: {0}'.format(np.mean(1. / np.array(ranks_right))))
    logger.info('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))

    output_eval_file = os.path.join(args.output_dir, file_prefix + "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logging.info("***** Eval results at {}*****".format(global_step))
        writer.write("***** Eval results at {}*****\n".format(global_step))
        for i in [0, 2, 9]:
            writer.write('Hits left @{0}: {1}\n'.format(i + 1, np.mean(hits_left[i])))
            writer.write('Hits right @{0}: {1}\n'.format(i + 1, np.mean(hits_right[i])))
            writer.write('Hits @{0}: {1}\n'.format(i + 1, np.mean(hits[i])))
        writer.write('Mean rank left: {0}\n'.format(np.mean(ranks_left)))
        writer.write('Mean rank right: {0}\n'.format(np.mean(ranks_right)))
        writer.write('Mean rank: {0}\n'.format(np.mean(ranks)))
        writer.write('Mean reciprocal rank left: {0}\n'.format(np.mean(1. / np.array(ranks_left))))
        writer.write('Mean reciprocal rank right: {0}\n'.format(np.mean(1. / np.array(ranks_right))))
        writer.write('Mean reciprocal rank: {0}\n'.format(np.mean(1. / np.array(ranks))))
        writer.write("\n")
    return np.mean(hits[9])

def train(args, train_dataset, model, tokenizer, eval_dataset=None, eval_fn=evaluate):
    eval_fn = eval_fn
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))

    # learning setup
    train_dataloader = setup_training_step(
        args, train_dataset, collate_fn=train_dataset.data_collate_fn, num_workers=args.num_workers)

    # Prepare optimizer and schedule (linear warmup and decay)
    model, optimizer, scheduler = setup_opt(args, model)
    metric_best = -1e5
    global_step = 0


    ma_dict = MovingAverageDict()
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _idx_epoch, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader,
                              desc="Iteration-{}({})".format(_idx_epoch, args.gradient_accumulation_steps),
                              disable=args.local_rank not in [-1, 0])
        step_loss = 0
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = train_dataset.batch2feed_dict(batch)

            outputs = model(**inputs)

            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            loss = update_wrt_loss(args, model, optimizer, loss)
            step_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                model_update_wrt_gradient(args, model, optimizer, scheduler)

                global_step += 1

                # update loss for logging
                if args.local_rank in [-1, 0]:
                    tb_writer.add_scalar("training_loss", step_loss, global_step)
                ma_dict({"loss": step_loss})
                step_loss = 0.

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging.info(ma_dict.get_val_str())

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_model_with_default_name(args.output_dir, model, tokenizer, args)

                if args.local_rank in [-1, 0] and eval_dataset is not None \
                        and args.eval_steps > 0 and global_step % args.eval_steps == 0 :
                    metric_cur = eval_fn(
                        args, eval_dataset, model, tokenizer, global_step=global_step, file_prefix="eval_")
                    if metric_cur > metric_best:
                        save_model_with_default_name(args.output_dir, model, tokenizer, args)
                        metric_best = metric_cur
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        # can add epoch evaluation
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

