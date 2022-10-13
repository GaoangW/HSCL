import torch
import torch.distributed as dist
import diffdist.functional as distops
import numpy as np
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score


def get_similarity_matrix(outputs, chunk=2, multi_gpu=False):
    '''
        Compute similarity matrix
        - outputs: (B', d) tensor for B' = B * chunk
        - sim_matrix: (B', B') tensor
    '''

    if multi_gpu:
        outputs_gathered = []
        for out in outputs.chunk(chunk):
            gather_t = [torch.empty_like(out) for _ in range(dist.get_world_size())]
            gather_t = torch.cat(distops.all_gather(gather_t, out))
            outputs_gathered.append(gather_t)
        outputs = torch.cat(outputs_gathered)

    sim_matrix = torch.mm(outputs, outputs.t())  # (B', d), (d, B') -> (B', B')

    return sim_matrix


def NT_xent_v2(sim):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''
    r1 = 16
    r2 = 16
    eps = 1e-8
    device = sim.device

    N_size = len(sim)//2
    eye = torch.eye(len(sim), device=device)
    eye_block = torch.eye(N_size, device=device)
    ones_block = torch.ones(N_size, N_size, device=device)
    pos_scale = r1*torch.cat([torch.cat([1.-ones_block, eye_block], dim=1), torch.cat([eye_block, 1.-ones_block], dim=1)], dim=0)
    neg_scale = r2*torch.cat([torch.cat([ones_block, 1.-eye_block], dim=1), torch.cat([1.-eye_block, ones_block], dim=1)], dim=0)
    scales = pos_scale+neg_scale

    sim = torch.exp(sim*scales)*(1 - eye)

    denom = torch.sum(sim, dim=1, keepdim=True)
    sim = -torch.log(sim / (denom + eps) + eps)  # loss matrix

    loss = torch.sum(sim[:N_size, N_size:].diag() + sim[N_size:, :N_size].diag()) / (2 * N_size)

    return loss

def NT_xent(sim_matrix, temperature=0.5, chunk=2, eps=1e-8):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''

    device = sim_matrix.device

    B = sim_matrix.size(0) // chunk  # B = B' / chunk

    eye = torch.eye(B * chunk).to(device)  # (B', B')
    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)  # loss matrix

    loss = torch.sum(sim_matrix[:B, B:].diag() + sim_matrix[B:, :B].diag()) / (2 * B)

    return loss


def Supervised_NT_xent(sim_matrix, labels, temperature=0.5, chunk=2, eps=1e-8, multi_gpu=False):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''

    device = sim_matrix.device

    if multi_gpu:
        gather_t = [torch.empty_like(labels) for _ in range(dist.get_world_size())]
        labels = torch.cat(distops.all_gather(gather_t, labels))
    labels = labels.repeat(2)

    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    sim_matrix = sim_matrix - logits_max.detach()

    B = sim_matrix.size(0) // chunk  # B = B' / chunk

    eye = torch.eye(B * chunk).to(device)  # (B', B')
    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)  # loss matrix

    labels = labels.contiguous().view(-1, 1)
    Mask = torch.eq(labels, labels.t()).float().to(device)
    Mask = Mask / (Mask.sum(dim=1, keepdim=True) + eps)

    loss = torch.sum(Mask * sim_matrix) / (2 * B)

    return loss
    


def convex_contrastive_with_polluted(P, model, images, labels, cur_epoch):

    class_idx = P.one_class_idx
    N_class = P.n_classes
    classes = P.classes

    # modify labels to binary, 0: normal unlabeled, 1: abnormal unlabeled, 2: normal labeled, 3: abnormal labeled
    mod_labels = labels.detach().clone()
    for n in range(len(mod_labels)):
        if mod_labels[n]<N_class:
            if mod_labels[n] in classes:
                mod_labels[n] = 0
            else:
                mod_labels[n] = 1
        else:
            if mod_labels[n]-N_class in classes:
                mod_labels[n] = 2
            else:
                mod_labels[n] = 3

    if P.multi_gpu:
        normal_vec = model.module.normal_vec
    else:
        normal_vec = model.normal_vec
    normal_vec = F.normalize(normal_vec, dim=0)

    k = 2
    r = 2 
    eps = 1e-8
    nv_pseudo_start_epoch = 0
    detach_flag = True
    neg_detach_flag = True

    w_thresh = 0.4 

    sort_top_k = 3
    pseudo_pos_ratio = 0.8

    B = len(images)
    loss = 0.
    device = images.device

    # check if abnormal data exists
    if torch.sum(mod_labels==3)==0:
        return loss

    # check if normal data exists
    if torch.sum(mod_labels==2)==0:
        return loss

    N_batch = len(images)//len(labels)
    for n in range(N_batch):
        if n==0:
            new_labels = mod_labels.detach().clone()
        else:
            new_labels = torch.cat([new_labels, mod_labels], dim=0)

    pos_label_idx = torch.nonzero(new_labels==2)
    neg_label_idx = torch.nonzero(new_labels==3)
    unlabeled_idx = torch.nonzero(new_labels<=1)

    pos_images = images[pos_label_idx[:,0]]
    neg_images = images[neg_label_idx[:,0]]
    unlabeled_images = images[unlabeled_idx[:,0]]
    N_pos = len(pos_images)
    N_neg = len(neg_images)
    N_unlabel = len(unlabeled_images)


    loss_vec = 0.
    
    norm_pos_images = F.normalize(pos_images)
    norm_neg_images = F.normalize(neg_images)
    norm_unlabeled_images = F.normalize(unlabeled_images)

 
    if detach_flag:
        A = torch.cat([norm_pos_images, norm_unlabeled_images], 0).detach()
    else:
        A = torch.cat([norm_pos_images, norm_unlabeled_images], 0)
            
    if neg_detach_flag:
        A2 = norm_neg_images.detach()
    else:
        A2 = norm_neg_images

    b = torch.ones(N_pos+N_unlabel, 1, device=pos_images.device)
    b2 = torch.ones(1, normal_vec.shape[1], device=pos_images.device)

    pos_unlabel_sim = (torch.max(torch.mm(norm_unlabeled_images, normal_vec), dim=1, keepdim=True)[0].detach()+1.)/2.
        
    w = torch.ones(N_pos+N_unlabel, 1, device=pos_images.device)
    w[N_pos:] = pos_unlabel_sim

        
    loss_vec = torch.mean(torch.square(w.detach()*(torch.max(torch.mm(A, normal_vec), dim=1, keepdim=True)[0]-b)))
    loss_vec += torch.mean(torch.square(torch.mm(A2, normal_vec)+1.))
        
    sim_mat = w.detach()*torch.mm(A, normal_vec)
    max_idx = torch.max(sim_mat, dim=0)[1].detach()
        

    comb_score_detach = w[N_pos:,0].detach()
    unlabeled_labels = new_labels[unlabeled_idx[:,0]]
    unlabeled_labels = 1-unlabeled_labels
        

    pos_images = torch.cat([pos_images, unlabeled_images], 0)
    N_pos = len(pos_images)
    w = w.detach().cpu().to(pos_images.device)
    w[w[:,0]<=w_thresh, 0] = 0.
    

    anchor_idx = torch.multinomial(w[:,0], B*k, replacement=True)
    anchor_idx = anchor_idx.view(B, k).to(device)
    anchor_w = torch.rand(B, k, device=device)
    anchor_w = anchor_w/torch.sum(anchor_w, dim=1, keepdim=True)
    anchor_vec = pos_images[anchor_idx.view(-1)]*anchor_w.view(-1).unsqueeze(1)
    anchor_vec = anchor_vec.view(B, k, images.shape[1])
    anchor_vec = torch.sum(anchor_vec, dim=1)
    anchor_vec = F.normalize(anchor_vec)

    pos_idx = torch.multinomial(w[:,0], B*k, replacement=True)
    pos_idx = pos_idx.view(B, k).to(device)
    #pos_idx = torch.randint(N_pos, (B, k), device=device)
    pos_w = torch.rand(B, k, device=device)
    pos_w = pos_w/torch.sum(pos_w, dim=1, keepdim=True)
    pos_vec = pos_images[pos_idx.view(-1)]*pos_w.view(-1).unsqueeze(1)
    pos_vec = pos_vec.view(B, k, images.shape[1])
    pos_vec = torch.sum(pos_vec, dim=1)
    pos_vec = F.normalize(pos_vec)

    
    neg_idx = torch.randint(N_neg, (B, k), device=device)
    neg_w = torch.rand(B, k, device=device)
    neg_w = neg_w/torch.sum(neg_w, dim=1, keepdim=True)
    neg_vec = neg_images[neg_idx.view(-1)]*neg_w.view(-1).unsqueeze(1)
    neg_vec = neg_vec.view(B, k, images.shape[1])
    neg_vec = torch.sum(neg_vec, dim=1)
    neg_vec = F.normalize(neg_vec)


    pos_sim = torch.mm(anchor_vec, pos_vec.permute(1,0)).diag().unsqueeze(1)
    neg_sim = torch.mm(anchor_vec, neg_vec.permute(1,0)).diag().unsqueeze(1)
    

    sim = torch.cat([pos_sim, neg_sim], dim=1)*r
    pred = F.softmax(sim, dim=1)
    pred = -torch.log(pred[:,0])
    loss = torch.mean(pred)
    loss += loss_vec

    return loss


def get_NCE_loss(pos_vec, neg_vec, norm_vec, r):
    pos_sim = torch.max(torch.mm(pos_vec, norm_vec), dim=1)[0]
    neg_sim = torch.max(torch.mm(neg_vec, norm_vec), dim=1)[0]
    #sim = torch.cat([torch.unsqueeze(pos_sim, 1), neg_sim.repeat(len(pos_vec), 1)], dim=1)*r
    sim = torch.cat([torch.unsqueeze(pos_sim, 1), torch.unsqueeze(neg_sim, 1)], dim=1)*r
    #import pdb; pdb.set_trace()
    pred = F.softmax(sim, dim=1)
    pred = -torch.log(pred[:,0])
    loss = torch.mean(pred)
    return loss
