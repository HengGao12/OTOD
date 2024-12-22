#!/bin/bash
# sh scripts/ood/kl_matching/cifar100_test_ood_kl_matching.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood
export CUDA_VISIBLE_DEVICES='0'
PYTHONPATH='.':$PYTHONPATH \
# srun -p mediasuper -x SZ-IDC1-10-112-2-17 --gres=gpu:${GPU} \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} \

python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/wrn.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/klm.yml \
    --num_workers 8 \
    --network.checkpoint 'results/cifar10_wrn_base_e100_lr0.1_default/s0/best.ckpt'
    # results/cifar10_wrn_base_e100_lr0.1_default/s0/best.ckpt
    # results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt