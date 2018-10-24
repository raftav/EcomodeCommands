#!/bin/bash

# EXPERIMENT NUMBER
exp_num=1

# restore epoch
re=108

source activate tensorflow1.2_py2.7

CUDA_VISIBLE_DEVICES=4 python -u skip_gru_evaluation.py $exp_num $re > 'test_exp'$exp_num'_epoch'$re'.out' 2>'test_exp'$exp_num'_epoch'$re'.err' &

source deactivate
disown
