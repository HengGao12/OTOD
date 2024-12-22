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
    --config configs/datasets/cifar10/cifar10_224x224.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/vit.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/ebo.yml \
    --network.checkpoint 'results/pytorch_model.bin' \
    --num_workers 8 \
    --merge_option merge