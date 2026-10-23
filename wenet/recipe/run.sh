#!/bin/bash


# Copyright 2019 Mobvoi Inc. All Rights Reserved.
. ./path.sh || exit 1

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
stage=1
stop_stage=2

# You should change the following two parameters for multiple machine training,
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
# Force both rendezvous (Gloo/c10d store) and NCCL to use loopback
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
# see https://pytorch.org/docs/stable/elastic/run.html
HOST_NODE_ADDR="localhost:0"
num_nodes=1

train_engine=torch_ddp

deepspeed_config=../../aishell/s0/conf/ds_stage2.json
deepspeed_save_states="model_only"

# Relative to the recipe directory.
data_dir=data
main_dir=.
. ./tools/parse_options.sh || exit 1

nj=16

# data_type can be `raw` or `shard`. Typically, raw is used for small dataset,
# `shard` is used for large dataset which is over 1k hours, and `shard` is
# faster on reading data and training.
data_type=raw
num_utts_per_shard=1000

train_set=train
# Optional train_config
# 1. conf/train_transformer.yaml: Standard transformer
# 2. conf/train_conformer.yaml: Standard conformer
# 3. conf/train_unified_conformer.yaml: Unified dynamic chunk causal conformer
# 4. conf/train_unified_transformer.yaml: Unified dynamic chunk transformer
# 5. conf/train_conformer_no_pos.yaml: Conformer without relative positional encoding
# 6. conf/train_u2++_conformer.yaml: U2++ conformer
# 7. conf/train_u2++_transformer.yaml: U2++ transformer
train_config=$main_dir/conf/train_u2++_conformer.yaml
cmvn=true
dir=$main_dir/exp/conformer
# Set checkpoint to continue
# checkpoint=


nbpe=1000
# nbpe=100

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
average_num=20
decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring"
# decode_modes="attention attention_rescoring"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: compute global cmvn"
    # compute cmvn
    python tools/compute_cmvn_stats.py --num_workers 16 --train_config $train_config \
        --in_scp ${data_dir}/${train_set}/wav.scp \
        --out_cmvn ${data_dir}/${train_set}/global_cmvn
    echo "Finish stage 1"
fi


bpemode=unigram
dict=${main_dir}/lang_char_/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=${main_dir}/lang_char_/${train_set}_${bpemode}${nbpe}
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  ### Task dependent. You have to check non-linguistic symbols used in the corpus.
  echo "stage 2: Dictionary and Json Data Preparation"
  mkdir -p ${main_dir}/lang_char_/
  echo "<blank> 0" > ${dict} # 0 will be used for "blank" in CTC
  echo "<unk> 1" >> ${dict} # <unk> must be 1
  # echo "<sos/eos> 2" >> $dict # <eos>

  # we borrowed these code and scripts which are related bpe from ESPnet.
  cut -f 2- -d" " ${data_dir}/${train_set}/text > ${main_dir}/lang_char_/input.txt
  tools/spm_train --input=${main_dir}/lang_char_/input.txt --vocab_size=${nbpe} \
    --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
  tools/spm_encode --model=${bpemodel}.model --output_format=piece \
    < ${main_dir}/lang_char_/input.txt | \
    tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
  
  num_token=$(cat $dict | wc -l)
  echo "<sos/eos> $num_token" >> $dict # <eos>
  wc -l ${dict}
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "stage 3: Prepare data, prepare required format"
  for x in dev test ${train_set}; do
    if [ $data_type == "shard" ]; then
      python tools/make_shard_list.py --num_utts_per_shard $num_utts_per_shard \
        --num_threads 16 ${data_dir}/$x/wav.scp ${data_dir}/$x/text \
        $(realpath ${data_dir}/$x/shards) ${data_dir}/$x/data.list
    else
      python tools/make_raw_list.py ${data_dir}/$x/wav.scp ${data_dir}/$x/text \
        ${data_dir}/$x/data.list
    fi
  done
  echo "Finish stage 3"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  mkdir -p $dir
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="nccl"
  # cmvn_opts=
  # $cmvn && cp ${data_dir}/${train_set}/global_cmvn $dir
  # $cmvn && cmvn_opts="--cmvn ${dir}/global_cmvn"

  # train.py rewrite $train_config to $dir/train.yaml with model input
  # and output dimension, and $dir/train.yaml will be used for inference
  # and export.
  if [ ${train_engine} == "deepspeed" ]; then
    echo "$0: using deepspeed"
  else
    echo "$0: using torch ddp"
  fi
  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus --rdzv_endpoint=$HOST_NODE_ADDR \
           --rdzv_id=2023 --rdzv_backend="c10d" \
    wenet/bin/train.py \
      --train_engine ${train_engine} \
      --config $train_config \
      --data_type $data_type \
      --train_data "${data_dir}/${train_set}/data.list" \
      --cv_data "${data_dir}/dev/data.list" \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --ddp.dist_backend $dist_backend \
      --num_workers 1 \
      --pin_memory \
      --deepspeed_config ${deepspeed_config} \
      --deepspeed.save_states ${deepspeed_save_states}
fi

