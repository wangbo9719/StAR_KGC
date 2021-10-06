import argparse
import collections
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from ensemble.ensemble_model import EnsembleModel
from ensemble.ensemble_dataset import EnsembleDataset, KbDataset
from peach.help import *
from peach.common import StAR_FILE_PATH


def get_optimizer(args, model):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.adam_betas is not None:
        adam_betas = tuple(float(_f) for _f in args.adam_betas.split(","))
        assert len(adam_betas) == 2
    else:
        adam_betas = (0.9, 0.999)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate,
                      betas=adam_betas, eps=args.adam_epsilon)
    return optimizer
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--init", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--dataset_dir", default=None, type=str, required=True,
                        help="The dataset dir.")
    parser.add_argument("--neg_times", default=10, type=int)
    parser.add_argument("--mode", default=None, type=str, required=True)
    parser.add_argument("--test_batch_size", default=1024, type=int)
    parser.add_argument("--context_score_path", default=None, type=str)
    parser.add_argument("--translation_score_path", default=None, type=str)
    parser.add_argument("--feature_method", default=None, type=str)
    parser.add_argument("--dataset", default='WN18RR', type=str)

    parser.add_argument("--hinge_loss_margin", default=0.6, type=float)
    parser.add_argument("--alpha_method", default="cls", type=str)
    parser.add_argument("--get_adp_improve", action="store_true")
    parser.add_argument("--seen_feature", action="store_true")
    parser.add_argument("--unseen_ablation", action="store_true")
    # extra parameters for prediction
    parser.add_argument("--no_verbose", action="store_true")

    ## Other parameters
    define_hparams_training(parser)
    args = parser.parse_args()

    # setup
    setup_prerequisite(args)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Dataset


    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)
    if StAR_FILE_PATH is None:
        print("Please replace StAR_FILE_PATH in ./StAR/peach/common.py with your own path to run the code.")
        return
    kb_train_dataset = KbDataset(
        args.dataset, "train", None, StAR_FILE_PATH+"/StAR/data/")
    kb_dev_dataset = KbDataset(
        args.dataset, "dev", None, StAR_FILE_PATH+"/StAR/data/")
    kb_test_dataset = KbDataset(
        args.dataset, "test", None, StAR_FILE_PATH+"/StAR/data/")
    kb_dataset_list = [kb_train_dataset, kb_dev_dataset, kb_test_dataset]

    # get_g
    g_subj2objs = collections.defaultdict(lambda: collections.defaultdict(set))
    g_obj2subjs = collections.defaultdict(lambda: collections.defaultdict(set))

    # prepare to remove the true triples
    for _ds in kb_dataset_list:
        for _raw_ex in _ds.raw_examples:
            _head, _rel, _tail = _raw_ex
            g_subj2objs[_head][_rel].add(_tail)
            g_obj2subjs[_tail][_rel].add(_head)


    # for _ep in [3, 5, 10]:
    #
    # for _lr in [1e-3, 1e-4, 1e-5]:
    #     for _nt in [5, 10]:
    #         for _margin in [0.6, 0.57, 0.63]:
    #             for _bs in [32, 64, 128]:
    # args.learning_rate = _lr
    # args.num_train_epochs = 3
    # args.neg_times = _nt
    # args.train_batch_size = _bs
    # args.hinge_loss_margin = _margin
    train_dataset = EnsembleDataset('train', args.mode, args.dataset_dir, args.neg_times)
    model = EnsembleModel(args.dataset_dir, train_dataset.top_k, args.neg_times,
                          args.hinge_loss_margin)
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    optimizer = get_optimizer(args, model)
    if args.init:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init)
        checkpoint = torch.load(os.path.join(args.init, 'checkpoint_'+args.mode+'_'+args.feature_method+'_'+str(args.train_batch_size)+'_' + str(args.learning_rate) +
                                 '_' + str(args.hinge_loss_margin) +'_'+str(int(args.num_train_epochs)) +'_'+str(int(args.neg_times))))
        # checkpoint = torch.load(os.path.join(args.init, 'checkpoint_'+args.mode+'_'+args.feature_method+'_'+str(args.hinge_loss_margin) + '_' +str(args.learning_rate) +'_' + str(args.train_batch_size)+
        #                          '_' +str(int(args.num_train_epochs))))
        init_step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if args.do_train and not args.do_eval:
        model.train_step(model, optimizer, train_dataset, args, g_subj2objs=g_subj2objs, g_obj2subjs=g_obj2subjs)
    if args.do_train and args.do_eval:
        model.train_step(model, optimizer, train_dataset, args, kb_dev_dataset, model.test_step, kb_dataset_list,g_subj2objs, g_obj2subjs)
    if not args.do_train and args.do_eval:
        model.test_step(model, kb_dev_dataset, kb_dataset_list, args, data_type='dev',g_subj2objs=g_subj2objs, g_obj2subjs=g_obj2subjs)
    if args.do_prediction:
        model.test_step(model, kb_test_dataset, kb_dataset_list, args, data_type='test',g_subj2objs=g_subj2objs, g_obj2subjs=g_obj2subjs)


if __name__ == '__main__':
    main()

