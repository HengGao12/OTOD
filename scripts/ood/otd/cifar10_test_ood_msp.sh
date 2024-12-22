#!/bin/bash
# sh scripts/ood/msp/cifar100_test_ood_msp.sh

# GPU=1
# CPU=1
# node=36
# jobname=openood
export CUDA_VISIBLE_DEVICES='0'
PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/wrn.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/msp.yml \
    --num_workers 8 \
    --network.checkpoint 'results/cifar10_wrn_base_e100_lr0.1_default/s0/best.ckpt'
    # /public/home/gaoheng/gh_workspace/OAML/results/cifar10_wrn_base_e100_lr0.1_default/s0/best.ckpt
    # results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt