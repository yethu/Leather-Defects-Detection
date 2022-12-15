import numpy as np
import torch
import torch.nn.functional as F
from skimage.measure import label, regionprops
from skimage import morphology
from skimage.segmentation import mark_boundaries, clear_border
from segmentation_models.metrics import IOUScore 
from sklearn import metrics
import matplotlib.pyplot as plt
from model import *
import time

np.random.seed(0)
_GCONST_ = -0.9189385332046727 # ln(sqrt(2*pi))
gamma = 0.0
theta = torch.nn.Sigmoid()
log_theta = torch.nn.LogSigmoid()

def get_threshold(dataset,cfg):
    L = cfg.pool_layers # number of pooled layers
    P = cfg.condition_vec
    N = 256

    encoder, pool_layers, pool_dims = load_encoder_arch(cfg,L)
    encoder = encoder.cuda().eval()

    decoders = [load_decoder_arch(cfg, pool_dim) for pool_dim in pool_dims]
    decoders = [decoder.cuda() for decoder in decoders]
    params = list(decoders[0].parameters())

    for l in range(1, cfg.pool_layers):
        params += list(decoders[l].parameters())

    load_weights(encoder, decoders)

    height = list()
    width = list()
    image_list = list()
    test_dist = [list() for layer in pool_layers]

    with torch.no_grad():
        for i, sample in enumerate(dataset):
            image_list.extend(t2np(sample['image']))
            # data
            image = sample['image'].cuda()  # single scale
            image = image.float().cuda()

            _ = encoder(image)  # BxCxHxW
            # test decoder
            e_list = list()
            for l, layer in enumerate(pool_layers):
                e = activation[layer]  # BxCxHxW
                #
                B, C, H, W = e.size()
                S = H*W
                E = B*S
                #
                if i == 0:  # get stats
                    height.append(H)
                    width.append(W)
                #
                p = positionalencoding2d(P, H, W).cuda().unsqueeze(0).repeat(B, 1, 1, 1)
                c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
                #
                decoder = decoders[l]
                FIB = E//N + int(E%N > 0)  # number of fiber batches
                for f in range(FIB):
                    if f < (FIB-1):
                        idx = torch.arange(f*N, (f+1)*N)
                    else:
                        idx = torch.arange(f*N, E)
                    #
                    c_p = c_r[idx]  # NxP
                    e_p = e_r[idx]  # NxC
                    #
                    if 'cflow' in cfg.dec_arch:
                        z, log_jac_det = decoder(e_p, [c_p,])
                    else:
                        z, log_jac_det = decoder(e_p)
                    #
                    decoder_log_prob = get_logp(C, z, log_jac_det)
                    log_prob = decoder_log_prob / C  # likelihood per dim
                    test_dist[l] = test_dist[l] + log_prob.detach().cpu().tolist()
    #

    test_map = [list() for p in pool_layers]
    i=1
    for l, p in enumerate(pool_layers):
        test_norm = torch.tensor(test_dist[l], dtype=torch.double)  # EHWx1
        test_norm-= torch.max(test_norm) # normalize likelihoods to (-Inf:0] by subtracting a constant
        test_prob = torch.exp(test_norm) # convert to probs in range [0:1]
        test_mask = test_prob.reshape(-1, height[l], width[l])
        test_mask = test_prob.reshape(-1, height[l], width[l])
        
        # upsample
        test_map[l] = F.interpolate(test_mask.unsqueeze(1),
            size=cfg.crp_size, mode='bilinear', align_corners=True).squeeze().numpy()
    
    # score aggregation
    score_map = np.zeros_like(test_map[0])
    for l, p in enumerate(pool_layers):
        score_map += test_map[l]
   
    score_mask = score_map/len(pool_layers)

    # invert probs to anomaly scores
    super_mask = score_mask.max() - score_mask
            
    total_rec = np.array(super_mask)
    return float(np.percentile(total_rec, [99]))

def train_meta_epoch(c, epoch, loader, encoder, decoders, optimizer, pool_layers, N):
    P = c.condition_vec
    L = c.pool_layers
    decoders = [decoder.train() for decoder in decoders]
    adjust_learning_rate(c, optimizer, epoch)
    I = len(loader)
    iterator = iter(loader)
    for sub_epoch in range(c.sub_epochs):
        train_loss = 0.0
        train_count = 0
        for i in range(I):
            try:
                #image, _, _ = next(iterator)
                sample=next(iterator)
            except StopIteration:
                iterator = iter(loader)
                sample=next(iterator)
            # encoder prediction
            image = sample['image'].cuda() # single scale
            image = image.float().cuda()
            with torch.no_grad():
                _ = encoder(image)
            # train decoder
            e_list = list()
            c_list = list()
            for l, layer in enumerate(pool_layers):               
                e = activation[layer].detach()  # BxCxHxW
                #
                B, C, H, W = e.size()
                S = H*W
                E = B*S    
                #
                p = positionalencoding2d(P, H, W).to(0).unsqueeze(0).repeat(B, 1, 1, 1)
                c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
                perm = torch.randperm(E).to(0)  # BHW
                decoder = decoders[l]
                #
                FIB = E//N  # number of fiber batches
                assert FIB > 0, 'MAKE SURE WE HAVE ENOUGH FIBERS, otherwise decrease N or batch-size!'
                for f in range(FIB):  # per-fiber processing
                    idx = torch.arange(f*N, (f+1)*N)
                    c_p = c_r[perm[idx]]  # NxP
                    e_p = e_r[perm[idx]]  # NxC
                    if 'cflow' in c.dec_arch:
                        z, log_jac_det = decoder(e_p, [c_p,])
                    else:
                        z, log_jac_det = decoder(e_p)
                    #
                    decoder_log_prob = get_logp(C, z, log_jac_det)
                    log_prob = decoder_log_prob / C  # likelihood per dim
                    loss = -log_theta(log_prob)
                    optimizer.zero_grad()
                    loss.mean().backward()
                    optimizer.step()
                    train_loss += t2np(loss.sum())
                    train_count += len(loss)
        #
    return train_loss / train_count

def val_meta_epoch(c, epoch, val_data, encoder, decoders, optimizer, pool_layers, N):
    P = c.condition_vec
    L = c.pool_layers
    decoders = [decoder.train() for decoder in decoders]
    adjust_learning_rate(c, optimizer, epoch)
    I = len(val_data)
    iterator = iter(val_data)
    for sub_epoch in range(c.sub_epochs):
        val_loss = 0.0
        val_count=0
        for i in range(I):
            # warm-up learning rate
            lr = warmup_learning_rate(c, epoch, i+sub_epoch*I, I*c.sub_epochs, optimizer)
            # sample batch
            try:
                #image, _, _ = next(iterator)
                sample=next(iterator)
            except StopIteration:
                iterator = iter(val_data)
                sample=next(iterator)
            # encoder prediction
            image = sample['image'].cuda() # single scale
            image = image.float().cuda()
            with torch.no_grad():
                _ = encoder(image)
            # train decoder
            e_list = list()
            c_list = list()
            for l, layer in enumerate(pool_layers):
                e = activation[layer].detach()  # BxCxHxW
                #
                B, C, H, W = e.size()
                S = H*W
                E = B*S    
                #
                p = positionalencoding2d(P, H, W).cuda().unsqueeze(0).repeat(B, 1, 1, 1)
                c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
                perm = torch.randperm(E).cuda()  # BHW
                decoder = decoders[l]
                #
                FIB = E//N  # number of fiber batches
                assert FIB > 0, 'MAKE SURE WE HAVE ENOUGH FIBERS, otherwise decrease N or batch-size!'
                for f in range(FIB):  # per-fiber processing
                    idx = torch.arange(f*N, (f+1)*N)
                    c_p = c_r[perm[idx]]  # NxP
                    e_p = e_r[perm[idx]]  # NxC
                    if 'cflow' in c.dec_arch:
                        z, log_jac_det = decoder(e_p, [c_p,])
                    else:
                        z, log_jac_det = decoder(e_p)
                    #
                    decoder_log_prob = get_logp(C, z, log_jac_det)
                    log_prob = decoder_log_prob / C  # likelihood per dim
                    loss = -log_theta(log_prob)
                    val_loss += t2np(loss.sum())
                    val_count += len(loss)
        #
        mean_val_loss = val_loss / val_count
                   
    return mean_val_loss

def test_meta_epoch(c, loader, encoder, decoders, pool_layers, N):

    if c.verbose:
        print('\nCompute loss and scores on test set:')
    #
    P = c.condition_vec
    decoders = [decoder.eval() for decoder in decoders]
    height = list()
    width = list()
    image_list = list()
    gt_label_list = list()
    gt_mask_list = list()
    test_dist = [list() for layer in pool_layers]
    
    tempo = 0
    with torch.no_grad():
        for i, sample in enumerate(loader):
            image_list.extend(t2np(sample['image']))
            gt_label_list.extend(t2np(sample['label']))
            gt_mask_list.extend(t2np(sample['mask']))
            # data
            image = sample['image'].cuda()  # single scale
            image = image.float().cuda()
            mask = sample['mask'].cuda() # single scale
            mask = mask.float().cuda()

            start = time.time()
            # data
            image = image.to(0) # single scale
            _ = encoder(image)  # BxCxHxW
            for l, layer in enumerate(pool_layers):
                e = activation[layer]  # BxCxHxW
                #
                B, C, H, W = e.size()
                S = H*W
                E = B*S
                #
                if i == 0:  # get stats
                    height.append(H)
                    width.append(W)
                #
                p = positionalencoding2d(P, H, W).to(0).unsqueeze(0).repeat(B, 1, 1, 1)
                c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
                #
                m = F.interpolate(mask, size=(H, W), mode='nearest')
                m_r = m.reshape(B, 1, S).transpose(1, 2).reshape(E, 1)  # BHWx1
                #
                decoder = decoders[l]
                FIB = E//N + int(E%N > 0)  # number of fiber batches
                for f in range(FIB):
                    if f < (FIB-1):
                        idx = torch.arange(f*N, (f+1)*N)
                    else:
                        idx = torch.arange(f*N, E)
                    #
                    c_p = c_r[idx]  # NxP
                    e_p = e_r[idx]  # NxC

                    z, log_jac_det = decoder(e_p, [c_p,])

                    decoder_log_prob = get_logp(C, z, log_jac_det)
                    log_prob = decoder_log_prob / C  # likelihood per dim
                    test_dist[l] = test_dist[l] + log_prob.detach().cpu().tolist()
                    
            tempo = tempo + (time.time()-start)
    
    return height, width, image_list, test_dist, gt_label_list, gt_mask_list

def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None

def get_logp(C, z, logdet_J):
    logp = C * _GCONST_ - 0.5*torch.sum(z**2, 1) + logdet_J
    return logp

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())

def split_data(dataset):
    """Split data in train 70%, validation 15% and test 15%

    Parameters
    -----------
    dataset:  
    Returns
    -----------
    train_dataset, val_dataset, test_dataset
    """
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    test_size = int(0.5 * len(test_dataset))
    val_size = len(test_dataset) - test_size

    val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [val_size, test_size])

    return train_dataset, val_dataset, test_dataset

def split_data_testing(dataset):
    test_size = int(0.85 * len(dataset))
    threshold_size = len(dataset) - test_size
    
    test_dataset, threshold_dataset = torch.utils.data.random_split(dataset, [test_size, threshold_size])

    return test_dataset, threshold_dataset

def optime_threshold(loader,pool_layers,cfg):
    L = cfg.pool_layers # number of pooled layers
    N = 256

    encoder, pool_layers, pool_dims = load_encoder_arch(cfg,L)
    encoder = encoder.cuda().eval()

    decoders = [load_decoder_arch(cfg, pool_dim) for pool_dim in pool_dims]
    decoders = [decoder.cuda() for decoder in decoders]
    params = list(decoders[0].parameters())

    for l in range(1, cfg.pool_layers):
        params += list(decoders[l].parameters())

    load_weights(encoder, decoders)
    
    height, width, test_image_list, test_dist, gt_label_list, gt_mask_list = test_meta_epoch(cfg, loader, encoder, decoders, pool_layers, N)

    # PxEHW
    print('Heights/Widths', height, width)
    test_map = [list() for p in pool_layers]
    for l, p in enumerate(pool_layers):
        test_norm = torch.tensor(test_dist[l], dtype=torch.double)  # EHWx1
        test_norm-= torch.max(test_norm) # normalize likelihoods to (-Inf:0] by subtracting a constant
        test_prob = torch.exp(test_norm) # convert to probs in range [0:1]
        test_mask = test_prob.reshape(-1, height[l], width[l])
        test_mask = test_prob.reshape(-1, height[l], width[l])
        # upsample
        test_map[l] = F.interpolate(test_mask.unsqueeze(1),size=cfg.crp_size, mode='bilinear', align_corners=True).squeeze().numpy()
    # score aggregation
    score_map = np.zeros_like(test_map[0])
    for l, p in enumerate(pool_layers):
        score_map += test_map[l]
    
    score_mask = score_map/len(pool_layers)
    # invert probs to anomaly scores
    super_mask = score_mask.max() - score_mask

    score_label = np.max(super_mask, axis=(1, 2))
    gt_label = np.asarray(gt_label_list, dtype=np.bool)
    precision, recall, thresholds = metrics.precision_recall_curve(gt_label, score_label)
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    det_threshold = thresholds[np.argmax(f1)]

    true_masks = np.asarray(gt_mask_list, dtype=np.bool)
    residuals = np.asarray(super_mask)
    precision, recall, thresholds = metrics.precision_recall_curve(true_masks.flatten(),residuals.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    seg_threshold = thresholds[np.argmax(f1)]
    

    return det_threshold, seg_threshold

def compute_pro(super_mask,gt_mask,threshold):
    max_step = 1000
    expect_fpr = 0.3  # default 30%
    max_th = super_mask.max()
    min_th = super_mask.min()
    delta = (max_th - min_th) / max_step
    ious_mean = []
    ious_std = []
    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(super_mask, dtype=np.bool)

    binary_score_maps[super_mask <= threshold] = 0
    binary_score_maps[super_mask >  threshold] = 1
    pro = []  # per region overlap
    # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
    # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map 
    for i in range(len(binary_score_maps)):    # for i th image
        # pro (per region level)
        label_map = label(gt_mask[i], connectivity=2)
        props = regionprops(label_map)
        for prop in props:
            x_min, y_min, x_max, y_max = prop.bbox    # find the bounding box of an anomaly region 
            cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
            # cropped_mask = gt_mask[i][x_min:x_max, y_min:y_max]   # bug!
            cropped_mask = prop.filled_image    # corrected!
            intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
            pro.append(intersection / prop.area)
        # iou (per image level)
        intersection = np.logical_and(binary_score_maps[i], gt_mask[i]).astype(np.float32).sum()
        union = np.logical_or(binary_score_maps[i], gt_mask[i]).astype(np.float32).sum()
    # against steps and average metrics on the testing data
    return np.array(pro).mean()

def evaluate(y_trues,residuals,labels,threshold,det_threshold,seg_threshold):
    
    residuals=np.array(residuals)

    #predicted mask using training threshold
    predicted_mask = np.zeros_like(residuals)
    predicted_mask[residuals >  threshold] = 1.0
    predicted_mask=clear_border(predicted_mask)

    #predicted mask optimized for detection
    predicted_mask_detection = np.zeros_like(residuals)
    predicted_mask_detection[residuals >  det_threshold] = 1.0
    predicted_mask_detection=clear_border(predicted_mask_detection)
    

    #predicted mask optimized for segmentation
    predicted_mask_segmentation = np.zeros_like(residuals)
    predicted_mask_segmentation[residuals >  seg_threshold] = 1.0
    predicted_mask_segmentation=clear_border(predicted_mask_segmentation)

    #detection and segmentation using AUROC
    score_label = np.max(residuals, axis=(1, 2))
    gt_label = np.asarray(labels, dtype=np.bool)
    det_roc_auc = metrics.roc_auc_score(gt_label, score_label)
    
    gt_mask = np.squeeze(np.asarray(y_trues, dtype=np.bool), axis=1)
    seg_roc_auc = metrics.roc_auc_score(gt_mask.flatten(), residuals.flatten())

    #Iou training threshold and segmentation optimized threshold
    iou=IOUScore()

    seg_iou_training_threshold = iou(gt_mask, predicted_mask)
    seg_iou_optimized_threshold = iou(gt_mask, predicted_mask_segmentation)
  
    #detection f1 score and precision
    labels_training_threshold = [mask.max() for mask in predicted_mask]
    labels_detection_threshold = [mask.max() for mask in predicted_mask_detection]
    
    f1score_training_threshold=metrics.f1_score(gt_label,labels_training_threshold)
    f1score_detection_threshold=metrics.f1_score(gt_label,labels_detection_threshold)

    pro_training_threshold = compute_pro(residuals,gt_mask,threshold)  
    pro_segmentation_threshold =  compute_pro(residuals,gt_mask,seg_threshold)  
    return det_roc_auc,seg_roc_auc, seg_iou_training_threshold, seg_iou_optimized_threshold, f1score_training_threshold, f1score_detection_threshold,pro_training_threshold, pro_segmentation_threshold

def get_image(image,gt,super_mask,threshold,det_threshold,seg_threshold):
    kernel = morphology.disk(4)

    image = (image.transpose(1, 2, 0)* 255).astype(np.uint8)
    gt = np.squeeze((gt.transpose(1, 2, 0)* 255).astype(np.uint8))

    score_mask1 = np.zeros_like(super_mask)
    score_mask1[super_mask >  threshold] = 1.0
    score_mask1=clear_border(score_mask1)
    score_mask1 = morphology.opening(score_mask1, kernel)
    score_mask1 = (255.0*score_mask1).astype(np.uint8)

    score_mask2 = np.zeros_like(super_mask)
    score_mask2[super_mask >  det_threshold] = 1.0
    score_mask2=clear_border(score_mask2)
    score_mask2 = morphology.opening(score_mask2, kernel)
    score_mask2 = (255.0*score_mask2).astype(np.uint8)

    score_mask3 = np.zeros_like(super_mask)
    score_mask3[super_mask >  seg_threshold] = 1.0
    score_mask3=clear_border(score_mask3)
    score_mask3 = morphology.opening(score_mask3, kernel)
    score_mask3 = (255.0*score_mask3).astype(np.uint8)

    t1 = mark_boundaries(image, score_mask1, color=(1, 0, 0), mode='thick')
    t2 = mark_boundaries(image, score_mask2, color=(1, 0, 0), mode='thick')
    t3 = mark_boundaries(image, score_mask3, color=(1, 0, 0), mode='thick')

    return image, gt, super_mask,  t1,t2,t3
