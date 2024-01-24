# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DIR="/media/data/wq/imagenet/ILSVRC2012_img_train"
ARCH="alexnet"
LR=0.05
WD=-5
K=10000
WORKERS=12
EXP="/media/data/wq/deepcluster_exp/exp_pro2.1"
PYTHON="/home/wqq/anaconda3/envs/pytorch/bin/python"

mkdir -p ${EXP}

CUDA_VISIBLE_DEVICES=1,2,3 ${PYTHON} main_promoted2.1.py ${DIR} --exp ${EXP} --arch ${ARCH} \
  --lr ${LR} --wd ${WD} --k ${K} --sobel --verbose --workers ${WORKERS}
