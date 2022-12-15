import torch
from torch.utils.data import DataLoader
from options import Options
from data_loader import TrainDataset, TestDataset
from utils import *
from model import *
import wandb
import time
import matplotlib.pyplot as plt


def train_on_device(cfg):
    last_loss = 100
    N = 256
    dataset = TrainDataset(path=cfg.train_data_dir, image_shape=cfg.image_shape) 
    
    train_dataset, val_dataset, threshold_dataset = split_data(dataset)
    
    train_dataset = DataLoader(train_dataset, batch_size=cfg.batch_size,shuffle=True)
    val_dataset = DataLoader(val_dataset, batch_size=cfg.batch_size,shuffle=True)
    threshold_dataset = DataLoader(threshold_dataset, batch_size=cfg.batch_size,shuffle=True)

    L = cfg.pool_layers # number of pooled layers
    print('Number of pool layers =', L)
    encoder, pool_layers, pool_dims = load_encoder_arch(cfg,L)
    encoder = encoder.cuda().eval()
    # NF decoder
    decoders = [load_decoder_arch(cfg, pool_dim) for pool_dim in pool_dims]
    decoders = [decoder.cuda()for decoder in decoders]
    params = list(decoders[0].parameters())
    for l in range(1, L):
        params += list(decoders[l].parameters())
    
    optimizer = torch.optim.Adam(params, lr=cfg.lr)

    print('train loader length', len(train_dataset.dataset))
    print('train loader batches', len(train_dataset))

    n_epochs=0
    
    for epoch in range(cfg.meta_epochs):
        total_train_loss = 0
        total_validation_loss = 0
        n_epochs+=1

        total_train_loss=train_meta_epoch(cfg, epoch, train_dataset, encoder, decoders, optimizer, pool_layers, N)
        total_validation_loss=val_meta_epoch(cfg, epoch, train_dataset, encoder, decoders, optimizer, pool_layers, N)

        if total_validation_loss > last_loss:
            trigger_times += 1
            if trigger_times >= cfg.patience:
                print('Early stopping!\n Not improve during the last 20 epochs.')
                break
        else:
            trigger_times = 0
            #save model
            state = {'encoder_state_dict': encoder.state_dict(),
                     'decoder_state_dict': [decoder.state_dict() for decoder in decoders]
                    }
   
            torch.save(state, "model.pckl")
            
            print('epoch [{}/{}], loss:{:.4f}, val_loss:{:.4f}'.format(epoch+1, cfg.meta_epochs,total_train_loss, total_validation_loss)) 
            last_loss = total_validation_loss

    threshold=get_threshold(threshold_dataset,cfg)  

    return threshold

def test_on_device(cfg, threshold):
    image_shape = (cfg.image_size, cfg.image_size, 3)
    mask_shape =  (cfg.image_size, cfg.image_size, 1)
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

    dataset = TestDataset(cfg.test_data_dir,cfg.mask_data_dir, image_shape=image_shape, mask_shape=mask_shape) 
    test_loader, threshold_loader = split_data_testing(dataset)

    test_dataset = DataLoader(test_loader, batch_size=1,shuffle=True)
    threshold_dataset = DataLoader(threshold_loader, batch_size=1,shuffle=True)

    det_threshold, seg_threshold=optime_threshold(threshold_dataset,pool_layers,cfg)

    height, width, test_image_list, test_dist, gt_label_list, gt_mask_list = test_meta_epoch(cfg, test_dataset, encoder, decoders, pool_layers, N)

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

    det_roc_auc,seg_roc_auc, seg_iou_training_threshold, seg_iou_optimized_threshold, f1score_training_threshold, f1score_optimized_threshold,pro_training_threshold, pro_segmentation_threshold = evaluate(gt_mask_list,super_mask,gt_label_list,threshold,det_threshold,seg_threshold)

    print("detection auroc: ", det_roc_auc)
    print("detection f1 score training threshold: ", f1score_training_threshold)
    print("detection f1 score optimized threshold: ", f1score_optimized_threshold)
    print("segmentation auroc: ", seg_roc_auc)
    print("segmentation training threshold: ", seg_iou_training_threshold)
    print("segmentation optimized threshold: ", seg_iou_optimized_threshold)

    #cm is a function to convert gray images to viridis color map
    cm = plt.get_cmap('viridis')

    for image, mask, label, map in zip (test_image_list,gt_mask_list,gt_label_list,super_mask):

        image, gt, super_mask, t1,t2,t3 = get_image(image,mask,map,threshold,det_threshold,seg_threshold)
        super_mask=cm(super_mask)

        fig = plt.figure(figsize=(8, 8))
        fig.add_subplot(2, 3, 1)
        plt.imshow(image)
        fig.add_subplot(2, 3, 2)
        plt.imshow(gt)
        fig.add_subplot(2, 3, 3)
        plt.imshow(super_mask)
        fig.add_subplot(2, 3, 4)
        plt.imshow(t1)
        fig.add_subplot(2, 3, 5)
        plt.imshow(t2)
        fig.add_subplot(2, 3, 6)
        plt.imshow(t3)
        plt.show()
        
# parse argument variables
cfg = Options().parse()
cfg.verbose = True 
cfg.condition_vec = 128
cfg.clamp_alpha = 1.9
cfg.lr_decay_rate = 0.1
cfg.lr_warm_epochs = 2
cfg.lr_warm = True
cfg.lr_cosine = True
cfg.hide_tqdm_bar = True

if cfg.lr_warm:
    cfg.lr_warmup_from = cfg.lr/10.0
    if cfg.lr_cosine:
        eta_min = cfg.lr * (cfg.lr_decay_rate ** 3)
        cfg.lr_warmup_to = eta_min + (cfg.lr - eta_min) * (1 + math.cos(math.pi * cfg.lr_warm_epochs / cfg.meta_epochs)) / 2
    else:
        cfg.lr_warmup_to = cfg.lr

cfg.crp_size = (cfg.image_size,cfg.image_size)  # HxW forma
cfg.image_shape = (cfg.image_size,cfg.image_size,3)

with torch.cuda.device(0):
    if(cfg.phase=="train"): 
        threshold=train_on_device(cfg)
    elif(cfg.phase=="test"):
        test_on_device(cfg, cfg.train_threshold)
    else:
        print("Insert train or test in --phase argument")
