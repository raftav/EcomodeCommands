#!/bin/bash

opt='adam'
exp_num=1

for lr in 0.0001; do
for batch in 1 5 10; do
for nu in 50 100 150; do
for nl in 3 5; do

source activate tf1.7_py3.5

CUDA_VISIBLE_DEVICES=0 python -u lstm_train.py $exp_num $lr $batch $opt $nu $nl > 'out_exp'$exp_num'.txt' 2>'out_exp'$exp_num'.err' &
pid=$!
wait $pid
disown
source deactivate

echo 'Pid = '$pid
echo 'Experiment number '$exp_num
echo 'Learning rate '$lr
echo 'Batch size '$batch
echo 'num hidden units '$nu
echo 'num hidden layers '$nl

exp_num=$(($exp_num+1))

done
done
done
done

