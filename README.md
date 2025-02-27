# Detecting OOD Samples via Optimal Transport Scoring Function

> Heng Gao, Zhuolin He, Jian Pu*  
> Fudan University

Before using our code, please first install the [OpenOOD](https://github.com/Jingkang50/OpenOOD). The pre-trained models are available [here](https://drive.google.com/drive/folders/1XSVB8pyYWuMVpq7BTuIESfU8GvCumxOn?usp=sharing). All our experiments are conducted on one NVIDIA V100 GPU.

### Data Preparation

Our codebase accesses the datasets from `./data/`.

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

## Citation
If you find **OTOD** useful in your research, please consider citing:
```
@misc{gao2024oalenhancingooddetection,
      title={Detecting OOD Samples via Optimal Transport Scoring Function}, 
      author={Heng Gao and Zhuolin He and Jian Pu},
      year={2025},
      eprint={2502.16115},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.16115}, 
}
```
