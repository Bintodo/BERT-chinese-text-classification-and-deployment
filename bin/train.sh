#!/usr/bin/env bash
export BERT_BASE_DIR=/home/zhangxingbin/workspace/bert_chainese/bert_model/chinese_L-12_H-768_A-12
export GLUE_DIR=/home/zhangxingbin/workspace/bert_chainese/data
export OUTPUT_DIR=/home/zhangxingbin/workspace/bert_chainese/output_01
export CUDA_VISIBLE_DEVICES=0,1

python /home/zhangxingbin/workspace/bert_chainese/src/classifier_gpu.py \
  --task_name=setiment \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/ \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=1024 \
  --learning_rate=2e-5 \
  --num_train_epochs=1.0 \
  --output_dir=$OUTPUT_DIR/

#--task_name=setiment
#--do_train=true
#--do_eval=true
#--data_dir=/home/zhangxingbin/workspace/bert_chainese/data/
#--vocab_file=/home/zhangxingbin/workspace/bert_chainese/bert_model/chinese_L-12_H-768_A-12/vocab.txt
#--bert_config_file=/home/zhangxingbin/workspace/bert_chainese/bert_model/chinese_L-12_H-768_A-12/bert_config.json
#--init_checkpoint=/home/zhangxingbin/workspace/bert_chainese/bert_model/chinese_L-12_H-768_A-12/bert_model.ckpt
#--max_seq_length=128
#--train_batch_size=2048
#--learning_rate=2e-5
#--num_train_epochs=1.0
#--output_dir=/home/zhangxingbin/workspace/bert_chainese/output_01/