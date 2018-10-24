#!/bin/bash

# EXPERIMENT NUMBER
exp_num=1

epoch=31

source activate tf1.7_py3.5

CUDA_VISIBLE_DEVICES=5 python -u lstm_test.py $exp_num $epoch > 'test_exp'$exp_num'_epoch'$epoch'.out' 2>'test_exp'$exp_num'_epoch'$epoch'.err' &

source deactivate
disown
