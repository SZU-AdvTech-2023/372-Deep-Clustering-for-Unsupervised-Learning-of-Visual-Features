
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DATA="/data/wq/imagenet/"
MODELROOT="/home/wq/projects/deepcluster"
MODEL="/data/wq/exp_pro2.1/checkpoint_100.0.pth.tar"
EXP="/data/wq/linear_pro2.1_c3"

PYTHON="/home/wq/anaconda3/envs/wq_pytorch/bin/python"

mkdir -p ${EXP}

CUDA_VISIBLE_DEVICES=3 ${PYTHON} eval_linear.py --model ${MODEL} --data ${DATA} --conv 3 --lr 0.01 \
  --wd -7 --tencrops --verbose --exp ${EXP} --workers 12
