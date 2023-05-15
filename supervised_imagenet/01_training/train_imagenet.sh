#!/bin/bash
FFCV_PATH=../data

train_func() {
    for sd in 0 10 100 1000 10000
    do
        CUDA_VISIBLE_DEVICES=0,1,2,3 python train_imagenet.py  \
            --config-file configs/rn50_40_epochs.yaml \
            --data.train_dataset=$FFCV_PATH/img_${1}_train.ffcv \
            --data.val_dataset=$FFCV_PATH/img_${1}_val.ffcv \
            --data.num_workers=12 --data.in_memory=1 \
            --dist.world_size=4 \
            --setting.seed=$sd \
            --setting.label_file="./configs/labels/imagenet_${1}.json" \
            --setting.tag="imagenet_${1}"
    done
}

# We need to create an FFCV object before running the training script below
train_func all

# for phyla in [ "3rd", "instr", "vert", "dog", "reptile", "bird", "invertebrate", "conveyance", "randomlab" ] ; do
#     train_func $phyla
# done
