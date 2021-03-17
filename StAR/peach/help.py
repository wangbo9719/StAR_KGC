# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import logging
import logging as logger
import os
import random
import math

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter


from transformers import AdamW, get_linear_schedule_with_warmup

# logger = logging.getLogger(__name__)
ALL_MODELS = []
# ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig,
#                                                                                 RobertaConfig, DistilBertConfig)), ())
#
# MODEL_CLASSES = {
#     'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
#     'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
#     'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
#     'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
#     'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
# }


def define_hparams_training(parser):
    ## Required parameters
    # parser.add_argument("--data_dir", default=None, type=str, required=True,
    #                     help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    # parser.add_argument("--model_type", default=None, type=str, required=True,
    #                     help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    # parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
    #                     help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
    #                         ALL_MODELS))
    # parser.add_argument("--task_name", default=None, type=str, required=True,
    #                     help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    # parser.add_argument("--output_dir", default=None, type=str, required=True,
    #                     help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_prediction", action='store_true',
                        help="Whether to run eval on the test set and save predictions")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")


    # parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
    #                     help="Batch size per GPU/CPU for training.")
    # parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
    #                     help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_betas", default=None, type=str,
                        help='betas for Adam optimizer')
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=-1, type=int,  # ! change to propostion
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=-1,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--eval_steps', type=int, default=-1,
                        help="Eval model every X updates steps. if X > 0")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--tpu', action='store_true',
                        help="Whether to run on the TPU defined in the environment variables")
    parser.add_argument('--tpu_ip_address', type=str, default='',
                        help="TPU IP address if none are set in the environment variables")
    parser.add_argument('--tpu_name', type=str, default='',
                        help="TPU name if none are set in the environment variables")
    parser.add_argument('--xrt_tpu_config', type=str, default='',
                        help="XRT TPU config if none are set in the environment variables")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")


def setup_logging(args):
    logger.basicConfig(format='%(asctime)s: %(message)s', level=logger.INFO, datefmt='%m/%d %I:%M:%S %p')
    # file_handler = logging.FileHandler(self.log_path)  # add a file handler to a logger
    # logging.getLogger().addHandler(file_handler)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def setup_prerequisite(args):
    # 1. output dir
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)  # Create output directory if needed

    # 2. Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # 3. Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # 4. setup TPU
    if args.tpu:
        if args.tpu_ip_address:
            os.environ["TPU_IP_ADDRESS"] = args.tpu_ip_address
        if args.tpu_name:
            os.environ["TPU_NAME"] = args.tpu_name
        if args.xrt_tpu_config:
            os.environ["XRT_TPU_CONFIG"] = args.xrt_tpu_config

        assert "TPU_IP_ADDRESS" in os.environ
        assert "TPU_NAME" in os.environ
        assert "XRT_TPU_CONFIG" in os.environ

        import torch_xla
        import torch_xla.core.xla_model as xm
        args.device = xm.xla_device()
        args.xla_model = xm

    # 5. setup logging
    setup_logging(args)

    # 6. Set Seed
    set_seed(args)

def setup_training_step_seq(args, train_dataset, **kwargs):
    assert args.train_batch_size % (args.n_gpu * args.gradient_accumulation_steps) == 0
    train_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    batch_size = args.train_batch_size // args.gradient_accumulation_steps
    if args.local_rank != -1:
        assert args.n_gpu == 1
        num_replicas = int(torch.distributed.get_world_size())
        assert args.train_batch_size % (num_replicas * args.gradient_accumulation_steps) == 0
        batch_size = args.train_batch_size // args.gradient_accumulation_steps // num_replicas

    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler,batch_size=batch_size, **kwargs)

    # learning step
    if args.max_steps <= 0:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    else:
        t_total = args.max_steps
        args.num_train_epochs = math.ceil(
            1. * args.max_steps * args.gradient_accumulation_steps / len(train_dataloader))

    if args.warmup_steps < 0:
        args.warmup_steps = math.ceil(t_total * args.warmup_proportion)

    args.t_total = t_total
    return train_dataloader

def setup_training_step(args, train_dataset, **kwargs):
    assert args.train_batch_size % (args.n_gpu * args.gradient_accumulation_steps) == 0
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    batch_size = args.train_batch_size // args.gradient_accumulation_steps
    if args.local_rank != -1:
        assert args.n_gpu == 1
        num_replicas = int(torch.distributed.get_world_size())
        assert args.train_batch_size % (num_replicas * args.gradient_accumulation_steps) == 0
        batch_size = args.train_batch_size // args.gradient_accumulation_steps // num_replicas

    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler,batch_size=batch_size, **kwargs)

    # learning step
    if args.max_steps <= 0:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    else:
        t_total = args.max_steps
        args.num_train_epochs = math.ceil(
            1. * args.max_steps * args.gradient_accumulation_steps / len(train_dataloader))

    if args.warmup_steps < 0:
        args.warmup_steps = math.ceil(t_total * args.warmup_proportion)

    args.t_total = t_total
    return train_dataloader


def setup_eval_step(args, eval_dataset, **kwargs):
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, **kwargs)
    return eval_dataloader


def setup_eval_model_for_fp16(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    return model


def setup_eval_model_for_inference(model, use_cuda, use_fp16, fp16_opt_level=None):
    device = torch.device("cpu")
    if use_cuda:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model.to(device)
        else:
            logging.warning("cuda is set to True but no cuda device available!")
        if use_fp16:
            optimizer = AdamW(model.parameters())
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)
    else:
        if use_fp16:
            logging.warning("cuda is set to False but fp16 is set to True -- omit!")

    return model, device





def setup_opt(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)
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
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    return model, optimizer, scheduler


def update_wrt_loss(args, model, optimizer, loss):
    if args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel training
    if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    return loss


def model_update_wrt_gradient(args, model, optimizer, scheduler):
    if args.max_grad_norm > 0.001:
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

    optimizer.step()
    scheduler.step()  # Update learning rate schedule
    model.zero_grad()


# =========================
# Others ==================
# =========================
def save_model_with_default_name(output_dir, model, tokenizer, args_to_save=None):
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info("Saving model checkpoint to %s", output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Good practice: save your training arguments together with the trained model
    if args_to_save is not None:
        torch.save(args_to_save, os.path.join(output_dir, 'training_args.bin'))


# ======

class MovingAverageDict(object):
    def __init__(self, decay=0.99):
        self.decay = decay
        self.ma_dict = {}

    def __call__(self, value_dict):
        for key, val in value_dict.items():
            if isinstance(val, (np.float32, np.float64, np.float16)) or \
                    (isinstance(val, np.ndarray) and val.dtype == "float32" and val.ndim == 0):
                val = float(val)

            if isinstance(val, float):
                if key not in self.ma_dict:
                    self.ma_dict[key] = MovingAverage()
                    # ma_dict的各个键对应的值是一个MovingAverage类
                self.ma_dict[key](val)  # 设定类的self.value,若不是新初始化的则value回进行衰减

    def get_val_dict(self):
        dict_return = {}
        for key, ma_obj in self.ma_dict.items():
            dict_return[key] = ma_obj.value
        return dict_return

    def get_val_str(self):
        val_dict = self.get_val_dict()
        # sort
        sorted_list = list(sorted(val_dict.items(), key=lambda item: item[0]))
        str_return = ""
        for key, val in sorted_list:
            if len(str_return) > 0:
                str_return += ", "
            str_return += "%s: %.4f" % (key, val)
        return str_return


class MovingAverage(object):
    def __init__(self, decay=0.99):
        self.decay = decay
        self.value = None

    def __call__(self, new_val):
        if self.value is None:
            self.value = new_val
        else:
            self.value = self.decay * self.value + (1. - self.decay) * new_val
        return self.value


def get_truncated_seqlen_list(seq_list, budge_len, ratio_list=None):
    ratio_list = ratio_list or ([1.] * len(seq_list))
    assert len(seq_list) > 0
    assert len(seq_list) == len(ratio_list)
    assert budge_len >= len(seq_list)

    lens_np = np.array([len(_e) for _e in seq_list], dtype="int64")
    ratios_np = np.array(ratio_list, dtype="float32")

    while sum(lens_np) > budge_len:
        lens_np[np.argmax(lens_np/ratios_np)] -= 1

    return list(lens_np)
