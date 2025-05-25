#!/bin/bash

expid=20
model_name=MTGNN
seed=3407
CUDA_VISIBLE_DEVICES=1
device='cuda:1'
runs=1

# train_forecaster=0
# if [ "$train_forecaster" == 1 ]; then
#     epochs=100
# else
#     epochs=0
# fi

for dataset in ECG5000 SOLAR TRAFFIC METR-LA
do
    step_size1=0
    mid_channels=64

    if [ "$dataset" == "ECG5000" ]; then
        step_size1=400
        mid_channels=64
    elif [ "$dataset" == "SOLAR" ]; then
        step_size1=2500
        mid_channels=512
    elif [ "$dataset" == "TRAFFIC" ]; then
        step_size1=1000
        mid_channels=256
    elif [ "$dataset" == "METR-LA" ]; then
        step_size1=2500
        mid_channels=128
    fi

    echo "forecaster: $model_name; dataset: $dataset; expid: $expid; step_size1: $step_size1; mid_channels: $mid_channels" 
    nohup python -u ../main_vida.py \
            --data '../data/'$dataset \
            --seed $seed \
            --model_name $model_name \
            --device $device \
            --expid $expid \
            --fc_epochs 100 \
            --da_epochs 100 \
            --batch_size 64 \
            --runs $runs \
            --random_node_idx_split_runs 100 \
            --w_fc 1.0 \
            --w_pc 0.5 \
            --w_align 0.5 \
            --patience 20 \
            --mid_channels $mid_channels \
            --useTCN True \
            --learning_rate 0.001 \
            --weight_decay 0.0001 \
            --step_size1 $step_size1 >../logs/main_exp/$dataset'_'$model_name'_exp'$expid'_seed'$seed'_mid_channels'$mid_channels'_v3_jl_ample'.out 2>&1 & 
    
done

echo "run success!"