python3 store_predictions.py +eval=default \
    eval.ckpt_dir=../checkpoints/imagenet_all

# for phyla in [ "3rd", "instr", "vert", "dog", "reptile", "bird", "invertebrate", "conveyance", "randomlab" ] ; do
#     python3 store_predictions.py +eval=default \
#         eval.ckpt_dir=../checkpoints/imagenet_$phyla
# done
