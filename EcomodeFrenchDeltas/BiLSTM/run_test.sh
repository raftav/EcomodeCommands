#!/bin/bash

# EXPERIMENT NUMBER
exp_num=1

epoch=17

source activate tensorflow1.2_py2.7

CUDA_VISIBLE_DEVICES=5 python -u lstm_test.py $exp_num $epoch >> 'test_exp'$exp_num'_epoch'$epoch'.out' 2>'test_exp'$exp_num'.err' &

source deactivate
disown
