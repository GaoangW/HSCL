import time

import torch.optim

import models.transform_layers as TL
from training.contrastive_loss import get_similarity_matrix, NT_xent, NT_xent_v2, convex_contrastive, convex_contrastive_with_polluted, unsup_convex_contrastive_with_polluted
from utils.utils import AverageMeter, normalize

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)


def train(P, epoch, model, criterion, optimizer, scheduler, loader, logger=None,
          simclr_aug=None, linear=None, linear_optim=None):

    assert simclr_aug is not None
    assert P.sim_lambda == 1.0  # to avoid mistake
    assert P.K_shift > 1

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['sim'] = AverageMeter()
    losses['shift'] = AverageMeter()

    check = time.time()
    for n, (images, labels) in enumerate(loader):
        model.train()
        count = n * P.n_gpus  # number of trained samples

        data_time.update(time.time() - check)
        check = time.time()

        ### SimCLR loss ###
        if P.dataset != 'imagenet':
            batch_size = images.size(0)
            images = images.to(device)
            images1, images2 = hflip(images.repeat(2, 1, 1, 1)).chunk(2)  # hflip
        else:
            batch_size = images[0].size(0)
            images1, images2 = images[0].to(device), images[1].to(device)
        labels = labels.to(device)

        images1 = torch.cat([P.shift_trans(images1, k) for k in range(P.K_shift)])
        images2 = torch.cat([P.shift_trans(images2, k) for k in range(P.K_shift)])
        shift_labels = torch.cat([torch.ones_like(labels) * k for k in range(P.K_shift)], 0)  # B -> 4B
        shift_labels = shift_labels.repeat(2)

        images_pair = torch.cat([images1, images2], dim=0)  # 8B
        images_pair = simclr_aug(images_pair)  # transform

        _, outputs_aux = model(images_pair, simclr=True, penultimate=True, shift=True)

        simclr = normalize(outputs_aux['simclr'])  # normalize
        sim_matrix = get_similarity_matrix(simclr, multi_gpu=P.multi_gpu)
        loss_sim = NT_xent(sim_matrix, temperature=0.5) * P.sim_lambda


        loss_shift = criterion(outputs_aux['shift'], shift_labels)

        
        if P.dataset=="cifar10_imagenet":
            convex_loss = convex_contrastive(P, model, simclr, labels, epoch)
        else:
            convex_loss = convex_contrastive_with_polluted(P, model, simclr, labels, epoch)
        
        
        ### total loss ###
        loss = loss_sim + loss_shift + convex_loss 

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #import pdb; pdb.set_trace()

        scheduler.step(epoch - 1 + n / len(loader))
        lr = optimizer.param_groups[0]['lr']

        batch_time.update(time.time() - check)

        ### Post-processing stuffs ###
        
        simclr_norm = outputs_aux['simclr'].norm(dim=1).mean()

        penul_1 = outputs_aux['penultimate'][:batch_size]
        penul_2 = outputs_aux['penultimate'][P.K_shift * batch_size: (P.K_shift + 1) * batch_size]
        outputs_aux['penultimate'] = torch.cat([penul_1, penul_2])  # only use original rotation

        ### Linear evaluation ###

        outputs_linear_eval = linear(outputs_aux['penultimate'].detach())


        new_labels = labels.repeat(2)
        if torch.sum(new_labels[new_labels==P.one_class_idx+P.n_classes])>0:
            loss_linear = criterion(outputs_linear_eval[new_labels==P.one_class_idx+P.n_classes], new_labels[new_labels==P.one_class_idx+P.n_classes]-P.n_classes)

            linear_optim.zero_grad()
            loss_linear.backward()
            linear_optim.step()
        else:
            linear_optim.zero_grad()
            linear_optim.step()

        losses['cls'].update(0, batch_size)
        losses['sim'].update(loss_sim.item(), batch_size)
        losses['shift'].update(loss_shift.item(), batch_size)

        if count % 50 == 0:
            log_('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                 '[LossC %f] [LossSim %f] [LossShift %f]' %
                 (epoch, count, batch_time.value, data_time.value, lr,
                  losses['cls'].value, losses['sim'].value, losses['shift'].value))

    log_('[DONE] [Time %.3f] [Data %.3f] [LossC %f] [LossSim %f] [LossShift %f]' %
         (batch_time.average, data_time.average,
          losses['cls'].average, losses['sim'].average, losses['shift'].average))

    if logger is not None:
        logger.scalar_summary('train/loss_cls', losses['cls'].average, epoch)
        logger.scalar_summary('train/loss_sim', losses['sim'].average, epoch)
        logger.scalar_summary('train/loss_shift', losses['shift'].average, epoch)
        logger.scalar_summary('train/batch_time', batch_time.average, epoch)

