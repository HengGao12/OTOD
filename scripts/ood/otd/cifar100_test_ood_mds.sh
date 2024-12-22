#!/bin/bash
# sh scripts/ood/mds/cifar100_test_ood_mds.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood
export CUDA_VISIBLE_DEVICES='0'
PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
    --config configs/datasets/cifar100/cifar100.yml \
    configs/datasets/cifar100/cifar100_ood.yml \
    configs/networks/wrn.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/mds.yml \
    --network.checkpoint 'results/cifar100_wrn_base_e100_lr0.1_default/s0/best.ckpt'
    # results/cifar100_wrn_base_e100_lr0.1_default/s0/best.ckpt
    # results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt