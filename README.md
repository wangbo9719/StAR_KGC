# STAR_KGC

This repo contains the source code of the paper accepted by WWW'2021. 
[**"Structure-Augmented Text Representation Learning for Efficient Knowledge Graph Completion"(WWW 2021).**](https://arxiv.org/abs/2004.14781)

## 1. Thanks
The repository is partially based on [huggingface transformers](https://github.com/huggingface/transformers), [KG-BERT](https://github.com/yao8839836/kg-bert) and [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding). 

## 2. Installing requirement packages
- conda create -n StAR python=3.6 
- source activate StAR
- pip install numpy torch tensorboardX tqdm boto3 requests regex sacremoses sentencepiece matplotlib

##### 2.1 Optional package (for mixed float Computation)
- git clone https://github.com/NVIDIA/apex
- cd apex
- pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

## 3. Dataset
- WN18RR, FB15k-237, UMLS 
	- Train and test set in ./data
	- As validation on original dev set is costly, we validated the model on dev subset during training. 
	- The dev subset of WN18RR is provided in ./data/WN18RR called *new_dev.dict*. Use below commands to get the dev subset for WN18RR (FB15k-237 is similar without the --do_lower_case) used in training process. 
	```
	CUDA_VISIBLE_DEVICES=0 \
	 python get_new_dev_dict.py \
		--model_class bert \
		--weight_decay 0.01 \
		--learning_rate 5e-5 \
		--adam_epsilon 1e-6 \
		--max_grad_norm 0. \
		--warmup_proportion 0.05 \
		--do_train \
		--num_train_epochs 7 \
		--dataset WN18RR \
		--max_seq_length 128 \
		--gradient_accumulation_steps 4 \
		--train_batch_size 16 \
		--eval_batch_size 128 \
		--logging_steps 100 \
		--eval_steps -1 \
		--save_steps 2000 \
		--model_name_or_path bert-base-uncased \
		--do_lower_case \
		--output_dir ./result/WN18RR_get_dev \
		--num_worker 12 \
		--seed 42 \
	```
	
	```
	CUDA_VISIBLE_DEVICES=0 \
	 python get_new_dev_dict.py \
		--model_class bert \
		--weight_decay 0.01 \
		--learning_rate 5e-5 \
		--adam_epsilon 1e-6 \
		--max_grad_norm 0. \
		--warmup_proportion 0.05 \
		--do_eval \
		--num_train_epochs 7 \
		--dataset WN18RR \
		--max_seq_length 128 \
		--gradient_accumulation_steps 4 \
		--train_batch_size 16 \
		--eval_batch_size 128 \
		--logging_steps 100 \
		--eval_steps 1000 \
		--save_steps 2000 \
		--model_name_or_path ./result/WN18RR_get_dev \
		--do_lower_case \
		--output_dir ./result/WN18RR_get_dev \
		--num_worker 12 \
		--seed 42 \
	```
- NELL-One 
    - We reformat original [NELL-One](https://github.com/xwhan/One-shot-Relational-Learning) as the three benchmarks above. 
    - Please run the below command to get the reformatted data.
	```
	python reformat_nell_one.py --data_dir path_to_downloaded --output_dir ./data/NELL_standard
	```
	
## 4. Training and Test (StAR)
Run the below commands for reproducing results in paper. Note, all the `eval_steps` is set to `-1` to train w/o validation and save the last checkpoint, because standard dev is very time-consuming. This can get similar results as in the paper. 

#### 4.1 WN18RR
```
CUDA_VISIBLE_DEVICES=0 \
python run_link_prediction.py \
    --model_class roberta \
    --weight_decay 0.01 \
    --learning_rate 1e-5 \
    --adam_betas 0.9,0.98 \
    --adam_epsilon 1e-6 \
    --max_grad_norm 0. \
    --warmup_proportion 0.05 \
    --do_train --do_eval \
    --do_prediction \
    --num_train_epochs 7 \
    --dataset WN18RR \
    --max_seq_length 128 \
    --gradient_accumulation_steps 4 \
    --train_batch_size 16 \
    --eval_batch_size 128 \
    --logging_steps 100 \
    --eval_steps 4000 \
    --save_steps 2000 \
    --model_name_or_path roberta-large \
    --output_dir ./result/WN18RR_roberta-large \
    --num_worker 12 \
    --seed 42 \
    --cls_method cls \
    --distance_metric euclidean \
```

#### 4.2 FB15k-237
```
CUDA_VISIBLE_DEVICES=0 \
python run_link_prediction.py \
    --model_class roberta \
    --weight_decay 0.01 \
    --learning_rate 1e-5 \
    --adam_betas 0.9,0.98 \
    --adam_epsilon 1e-6 \
    --max_grad_norm 0. \
    --warmup_proportion 0.05 \
    --do_train --do_eval \
    --do_prediction \
    --num_train_epochs 7. \
    --dataset FB15k-237 \
    --max_seq_length 100 \
    --gradient_accumulation_steps 4 \
    --train_batch_size 16 \
    --eval_batch_size 128 \
    --logging_steps 100 \
    --eval_steps -1 \
    --save_steps 2000 \
    --model_name_or_path roberta-large \
    --output_dir ./result/FB15k-237_roberta-large \
    --num_worker 12 \
    --seed 42 \
    --fp16 \
    --cls_method cls \
    --distance_metric euclidean \
```

#### 4.3 UMLS
```
CUDA_VISIBLE_DEVICES=0 \
python run_link_prediction.py \
    --model_class roberta \
    --weight_decay 0.01 \
    --learning_rate 1e-5 \
    --adam_betas 0.9,0.98 \
    --adam_epsilon 1e-6 \
    --max_grad_norm 0. \
    --warmup_proportion 0.05 \
    --do_train --do_eval \
    --do_prediction \
    --num_train_epochs 20 \
    --dataset UMLS \
    --max_seq_length 16 \
    --gradient_accumulation_steps 1 \
    --train_batch_size 16 \
    --eval_batch_size 128 \
    --logging_steps 100 \
    --eval_steps -1 \
    --save_steps 200 \
    --model_name_or_path roberta-large \
    --output_dir ./result/UMLS_model \
    --num_worker 12 \
    --seed 42 \
    --cls_method cls \
    --distance_metric euclidean 
```

#### 4.4 NELL-One
```
CUDA_VISIBLE_DEVICES=0 \
python run_link_prediction.py \
    --model_class bert \
    --do_train --do_eval \usepacka--do_prediction \
    --warmup_proportion 0.1 \
    --learning_rate 5e-5 \
    --num_train_epochs 8. \
    --dataset NELL_standard \
    --max_seq_length 32 \
    --gradient_accumulation_steps 1 \
    --train_batch_size 16 \
    --eval_batch_size 128 \
    --logging_steps 100 \
    --eval_steps -1 \
    --save_steps 2000 \
    --model_name_or_path bert-base-uncased \
    --do_lower_case \
    --output_dir ./result/NELL_model \
    --num_worker 12 \
    --seed 42 \
    --fp16 \
    --cls_method cls \
    --distance_metric euclidean 
```

## 5. StAR_Self-Adp

#### 5.1 Data preprocessing
- Get the trained model of RotatE, more details please refer to [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding).
- Run the below commands sequentially to get the training dataset of StAR_Self-Adp. 
	- Run the run_get_ensemble_data.py in ./StAR
	```
	CUDA_VISIBLE_DEVICES=0 python run_get_ensemble_data.py \
		--dataset WN18RR \
		--model_class roberta \
		--model_name_or_path ./result/WN18RR_roberta-large \
		--output_dir ./result/WN18RR_roberta-large \
		--seed 42 \
		--fp16 
	```
	
	- Run the ./codes/run.py in rotate. (please replace the TRAINED_MODEL_PATH with your own trained model's path)
	```
	CUDA_VISIBLE_DEVICES=2 python ./codes/run.py \
		--cuda --init ./models/RotatE_wn18rr_0 \
		--test_batch_size 16 \
		--star_info_path TRAINED_MODEL_PATH \
		--get_scores --get_model_dataset 
	```
	
#### 5.2 Train and Test
- Run the run.py in ./StAR/ensemble. Note the `--mode` should be alternate in `head` and `tail`, and perform a average operation to get the final results. 
- Note: Please replace YOUR_OUTPUT_DIR, TRAINED_MODEL_PATH and `StAR_FILE_PATH` in ./StAR/peach/common.py with your own paths to run the command and code.
```
CUDA_VISIBLE_DEVICES=0 python run.py \
--do_train --do_eval --do_prediction --seen_feature \
--mode tail \
--learning_rate 1e-3 \
--feature_method mix \
--neg_times 5 \
--num_train_epochs 3 \
--hinge_loss_margin 0.6 \
--train_batch_size 32 \
--test_batch_size 64 \
--logging_steps 100 \
--save_steps 2000 \
--eval_steps -1 \
--warmup_proportion 0 \
--output_dir YOUR_OUTPUT_DIR \
--dataset_dir TRAINED_MODEL_PATH \
--context_score_path TRAINED_MODEL_PATH \
--translation_score_path TRAINED_MODEL_PATH \
--seed 42 
```





