#!/bin/bash
IMAGENET_DIR=../data  # Read location of Imagenet train/val images
WRITE_DIR=../data     # Write location for the FFCV object

write_dataset_phyla () {
    write_path=$WRITE_DIR/img_${2}_${1}.ffcv
    echo "Writing ImageNet ${1} dataset to ${write_path}"

    python write_imagenet.py \
        --cfg.subset_file=./configs/labels/imagenet_${2}.json \
        --cfg.dataset=imagenet \
        --cfg.split=${1} \
        --cfg.data_dir=$IMAGENET_DIR/${1} \
        --cfg.write_path=$write_path \
        --cfg.max_resolution=500 \
        --cfg.write_mode=proportion \
        --cfg.compress_probability=0.50 \
        --cfg.jpeg_quality=90
}
 
#### Create subsets of Imagenet for FFCV
## We create a different FFCV object for each task which eats up 
## a lot of storage. There is definitely a more efficient way to 
## do this but we settled for this hacky approach.

write_dataset_phyla val   all
write_dataset_phyla train all

# for phyla in [ "3rd", "instr", "vert", "dog", "reptile", "bird", "invertebrate", "conveyance", "randomlab" ] ; do
#     write_dataset_phyla train $phyla
#     write_dataset_phyla val   $phyla
# done
