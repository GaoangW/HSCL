# Hierarchical Semi-Supervised Contrastive Learning for Contamination-Resistant Anomaly Detection

## Requirements
- python 3.6+
- torch 1.4+
- torchvision 0.5+
- CUDA 10.1+
- scikit-learn 0.22+
- tensorboard 2.0+
- [torchlars](https://github.com/kakaobrain/torchlars) == 0.1.2 
- [pytorch-gradual-warmup-lr](https://github.com/ildoonet/pytorch-gradual-warmup-lr) packages 
- [apex](https://github.com/NVIDIA/apex) == 0.1
- [diffdist](https://github.com/ag14774/diffdist) == 0.1 

## Training
### Train for class 0 on CIFAR-10
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py --dataset cifar10 --model resnet18 --mode simclr_CSI --shift_trans_type rotation --one_class_idx 0 --optimizer 'adam' --lr_init 0.001 --batch_size 256 --epochs 250 --pollute_ratio 0.05
```

## Testing
### Test for class 0 on CIFAR-10
```
python eval.py --mode ood_pre --dataset cifar10 --model resnet18 --ood_score CSI --shift_trans_type rotation --print_score --ood_samples 10 --resize_factor 0.54 --resize_fix --one_class_idx 0 --load_path "logs/cifar10_resnet18_unsup_simclr_CSI_shift_rotation_one_class_0/last.model" 
```
## Citation
```
@article{wang2022hierarchical,
  title={Hierarchical Semi-Supervised Contrastive Learning for Contamination-Resistant Anomaly Detection},
  author={Wang, Gaoang and Zhan, Yibing and Wang, Xinchao and Song, Mingli and Nahrstedt, Klara},
  journal={arXiv preprint arXiv:2207.11789},
  year={2022}
}
```

## Acknowledgement
The code structure is built on https://github.com/alinlab/CSI. We thank authors of CSI [1] to provide the source code and the solid work.

## Reference
[1] Tack, J., Mo, S., Jeong, J. and Shin, J., 2020. Csi: Novelty detection via contrastive learning on distributionally shifted instances. Advances in Neural Information Processing Systems, 2020.
