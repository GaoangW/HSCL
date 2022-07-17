from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from common.common import parse_args
import models.classifier as C
from datasets import get_dataset, get_superclass_list, get_subclass_dataset, get_subclass_with_abnormal_dataset, get_subclass_with_mixed_dataset
from utils.utils import load_checkpoint

P = parse_args()

### Set torch device ###

if torch.cuda.is_available():
    torch.cuda.set_device(P.local_rank)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

P.n_gpus = torch.cuda.device_count()

if P.n_gpus > 1:
    import apex
    import torch.distributed as dist
    from torch.utils.data.distributed import DistributedSampler

    P.multi_gpu = True
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
        world_size=P.n_gpus,
        rank=P.local_rank,
    )
else:
    P.multi_gpu = False

### only use one ood_layer while training
P.ood_layer = P.ood_layer[0]

### Initialize dataset ###
if P.dataset=="cifar10_imagenet":
    train_set, test_set, image_size, n_classes, cifar10_train_set, cifar10_test_set, ab_train, ab_test = get_dataset(P, dataset=P.dataset)
else:
    train_set, test_set, image_size, n_classes = get_dataset(P, dataset=P.dataset)
P.image_size = image_size
P.n_classes = n_classes

if P.one_class_idx is not None:
    cls_list = get_superclass_list(P.dataset)
    P.n_superclasses = len(cls_list)

    full_test_set = deepcopy(test_set)  # test set of full classes

    if not P.dataset=="cifar10_imagenet":

        if type(cls_list[P.one_class_idx])==int:
            choose_cls = [cls_list[P.one_class_idx]]
        else:
            choose_cls = cls_list[P.one_class_idx]
        test_set = get_subclass_dataset(test_set, classes=choose_cls)

        train_set = get_subclass_with_mixed_dataset(train_set, classes=choose_cls, normal_eta=0.05, polluted_eta=P.pollute_ratio, abnormal_eta=0.05)

        P.classes = choose_cls


kwargs = {'pin_memory': False, 'num_workers': 4}

if P.multi_gpu:
    train_sampler = DistributedSampler(train_set, num_replicas=P.n_gpus, rank=P.local_rank)
    test_sampler = DistributedSampler(test_set, num_replicas=P.n_gpus, rank=P.local_rank)
    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=P.batch_size, **kwargs)
    test_loader = DataLoader(test_set, sampler=test_sampler, batch_size=P.test_batch_size, **kwargs)
else:
    train_loader = DataLoader(train_set, shuffle=True, batch_size=P.batch_size, **kwargs)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)
#import pdb; pdb.set_trace()

if P.ood_dataset is None:
    if P.one_class_idx is not None:
        P.ood_dataset = list(range(P.n_superclasses))
        P.ood_dataset.pop(P.one_class_idx)
    elif P.dataset == 'cifar10':
        P.ood_dataset = ['svhn', 'lsun_resize', 'imagenet_resize', 'lsun_fix', 'imagenet_fix', 'cifar100', 'interp']
    elif P.dataset == 'imagenet':
        P.ood_dataset = ['cub', 'stanford_dogs', 'flowers102']

ood_test_loader = dict()
for ood in P.ood_dataset:
    if ood == 'interp':
        ood_test_loader[ood] = None  # dummy loader
        continue

    if P.one_class_idx is not None:
        if P.dataset=="cifar10_imagenet":
            ood_test_set = ab_test
        else:
            ood_test_set = get_subclass_dataset(full_test_set, classes=cls_list[ood])
        ood = f'one_class_{ood}'  # change save name
    else:
        ood_test_set = get_dataset(P, dataset=ood, test_only=True, image_size=P.image_size)

    if P.multi_gpu:
        ood_sampler = DistributedSampler(ood_test_set, num_replicas=P.n_gpus, rank=P.local_rank)
        ood_test_loader[ood] = DataLoader(ood_test_set, sampler=ood_sampler, batch_size=P.test_batch_size, **kwargs)
    else:
        ood_test_loader[ood] = DataLoader(ood_test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)

### Initialize model ###

simclr_aug = C.get_simclr_augmentation(P, image_size=P.image_size).to(device)
P.shift_trans, P.K_shift = C.get_shift_module(P, eval=True)
P.shift_trans = P.shift_trans.to(device)

model = C.get_classifier(P.model, n_classes=P.n_classes).to(device)
model = C.get_shift_classifer(model, P.K_shift).to(device)

criterion = nn.CrossEntropyLoss().to(device)

if P.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=P.lr_init, momentum=0.9, weight_decay=P.weight_decay)
    lr_decay_gamma = 0.1
elif P.optimizer == 'lars':
    from torchlars import LARS
    base_optimizer = optim.SGD(model.parameters(), lr=P.lr_init, momentum=0.9, weight_decay=P.weight_decay)
    optimizer = LARS(base_optimizer, eps=1e-8, trust_coef=0.001)
    lr_decay_gamma = 0.1
elif P.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=P.lr_init, weight_decay=P.weight_decay)
    lr_decay_gamma = 0.1
else:
    raise NotImplementedError()

if P.lr_scheduler == 'cosine':
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, P.epochs, eta_min=1e-6) #1e-6
elif P.lr_scheduler == 'step_decay':
    milestones = [int(0.5 * P.epochs), int(0.75 * P.epochs)]
    scheduler = lr_scheduler.MultiStepLR(optimizer, gamma=lr_decay_gamma, milestones=milestones)
else:
    raise NotImplementedError()

from training.scheduler import GradualWarmupScheduler
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=10.0, total_epoch=P.warmup, after_scheduler=scheduler)

if P.resume_path is not None:
    resume = True
    model_state, optim_state, config = load_checkpoint(P.resume_path, mode='last')
    model.load_state_dict(model_state, strict=not P.no_strict)
    optimizer.load_state_dict(optim_state)
    start_epoch = config['epoch']
    best = config['best']
    error = 100.0
else:
    resume = False
    start_epoch = 1
    best = 100.0
    error = 100.0

if P.mode == 'sup_linear' or P.mode == 'sup_CSI_linear':
    assert P.load_path is not None
    checkpoint = torch.load(P.load_path)
    model.load_state_dict(checkpoint, strict=not P.no_strict)

if P.multi_gpu:
    simclr_aug = apex.parallel.DistributedDataParallel(simclr_aug, delay_allreduce=True)
    model = apex.parallel.convert_syncbn_model(model)
    model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
