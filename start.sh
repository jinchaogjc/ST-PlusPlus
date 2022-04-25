#!/bin/bash 
export semi_setting='pascal/1_8/split_0'
CUDA_VISIBLE_DEVICE=S0,1 python -W ignore main.py \
    --dataset cityscapes --data-root ./data/cityscapes/ \
    --batch-size 16 --backbone resnet50 --model deeplabv3plus \
    --labeled-id-path dataset/splits/$semi_setting/labeled.txt \
    --unlabeled-id-path dataset/splits/$semi_setting/unlabeled.txt \
    --pseudo-mask-path outdir/pseudo_masks/$semi_setting \
    --save-path outdir/models/$semi_setting
