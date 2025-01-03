#!/bin/bash
# sh scripts/ood/she/cifar100_test_ood_she.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood
export CUDA_VISIBLE_DEVICES='1'
PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
    --config configs/datasets/cifar10/cifar10_224x224.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/vit.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/otdp.yml \
    --network.checkpoint 'results/pytorch_model.bin' \
    --merge_option merge
# python main.py \
#     --config configs/datasets/cifar100/cifar100.yml \
#     configs/datasets/cifar100/cifar100_ood.yml \
#     configs/networks/react_net.yml \
#     configs/pipelines/test/test_ood.yml \
#     configs/preprocessors/base_preprocessor.yml \
#     configs/postprocessors/otdp.yml \
#     --network.pretrained False \
#     --network.backbone.name resnet18_32x32 \
#     --network.backbone.pretrained True \
#     --network.backbone.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt' \
#     --num_workers 8 \
#     --merge_option merge
#  '/home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar100_resnet18_224x224_base_generative_ood_distill_trainer_e500_lr0.01_default_w_fea_n_logits_distill_4kl_8fd_0.1cl_0.2dcl_0.001ml_0.001dml/s0/best_epoch441_acc0.8050.ckpt'
# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar100_resnet18_224x224_base_ood_distill_trainer_e500_lr0.01_default_w_fea_n_logits_distill_4kl_8fd_0.1cl_for_ablation/s0/best_epoch383_acc0.8030.ckpt
# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar100_resnet18_224x224_base_ood_distill_trainer_e500_lr0.01_default_w_fea_n_logits_distill_4kl_8fd_0.1cl_for_ablation/s0/best_epoch383_acc0.8030.ckpt
# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar100_resnet18_224x224_distill_trainer_e100_lr0.01_default_w_logits_dist/s0/best_epoch93_acc0.7940.ckpt
# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar100_resnet18_224x224_base_distill_trainer_e500_lr0.01_default_w_fea_n_logits_distill_dev4_2kl_4fd/s0/best_epoch361_acc0.8090.ckpt
# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar100_resnet18_224x224_base_ood_distill_trainer_e500_lr0.01_default_w_fea_n_logits_distill_4kl_8fd_0.1cl_for_ablation/s0/best_epoch383_acc0.8030.ckpt
# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar100_resnet18_224x224_base_e100_lr0.1_default/s0/best.ckpt
# '/home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar100_resnet18_224x224_base_generative_ood_distill_trainer_e500_lr0.01_default_w_fea_n_logits_distill_4kl_8fd_0.1cl_0.2dcl_0.001ml_0.001dml/s0/best_epoch441_acc0.8050.ckpt'
# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar100_resnet18_224x224_base_e100_lr0.1_default/s0/best.ckpt
# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar100_resnet18_224x224_base_e100_lr0.1_default/s0/best.ckpt
# '/home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar100_resnet18_224x224_base_generative_ood_distill_trainer_e500_lr0.01_default_w_fea_n_logits_distill_4kl_8fd_0.1cl_0.2dcl_0.001ml_0.001dml/s0/best_epoch441_acc0.8050.ckpt'
# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar100_resnet18_224x224_base_generative_ood_distill_trainer_e500_lr0.01_default_w_fea_n_logits_distill_4kl_8fd_0.1cl_0.2dcl_0.001ml_0.001dml/s0/best_epoch441_acc0.8050.ckpt
# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar100_resnet18_224x224_base_e100_lr0.1_default/s0/best.ckpt
# /home1/gaoheng/gh_workspace/GOLDEN_HOOP/results/cifar100_resnet18_224x224_base_distill_trainer_e500_lr0.01_default_w_fea_n_logits_distill_dev4_2kl_4fd/s0/best_epoch361_acc0.8090.ckpt
############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs
# python scripts/eval_ood.py \
#     --id-data cifar100 \
#     --root ./results/cifar100_resnet18_32x32_base_e100_lr0.1_default \
#     --postprocessor gen \
#     --save-score --save-csv
