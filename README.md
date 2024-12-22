# Detecting OOD Samples via Optimal Transport Scoring Function

> Heng Gao, Zhuolin He, Jian Pu*  
> Fudan University

### Data Preparation

Our codebase accesses the datasets from `./data/` and pretrained models from `./results/checkpoints/` .

```
├── ...
├── data
│   ├── benchmark_imglist
│   ├── images_classic
├── openood
├── results
│   ├── checkpoints
│   └── ...
├── scripts
├── main.py
├── ...
```

### Usage

```sh
# CIFAR-100
bash scripts/ood/otd/cifar100_test_ood_otod.sh
# CIFAR-10
bash scripts/ood/otd/cifar10_test_ood_otod.sh
```
