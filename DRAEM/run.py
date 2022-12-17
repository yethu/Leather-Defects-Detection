from data_loader import MVTecDRAEMTrainDataset, MVTecDRAEMTestDataset
import torch
from torch.utils.data import DataLoader
from torch import optim
from config import Config
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from loss import FocalLoss, SSIM

import os
from utils import *


def get_threshold(dataset,model, model_seg,cfg):
    scores = []
    
    model.eval()
    model_seg.cuda()
    model_seg.eval()

    for batch in dataset:
        for image in batch["image"]:
            gray_batch = image.unsqueeze(0).cuda() 

            gray_rec = model(gray_batch)
            joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

            out_mask = model_seg(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)
            scores.append(out_mask_sm[0 ,1 ,: ,:].detach().cpu().numpy())
         
    scores = np.array(scores)

    return float(np.percentile(scores, [99]))
    
def train_one_step(model,model_seg, loss_l2, loss_ssim, loss_focal, scheduler, optimizer, data):
    gray_batch = data["image"].cuda()
    aug_gray_batch = data["augmented_image"].cuda()
    anomaly_mask = data["anomaly_mask"].cuda()

    gray_rec = model(aug_gray_batch)
    joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)

    out_mask = model_seg(joined_in)
    out_mask_sm = torch.softmax(out_mask, dim=1)

    l2_loss = loss_l2(gray_rec,gray_batch)
    ssim_loss = loss_ssim(gray_rec, gray_batch)

    segment_loss = loss_focal(out_mask_sm, anomaly_mask)
    loss = l2_loss + ssim_loss + segment_loss

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss

def validate_one_step(model,model_seg, loss_l2, loss_ssim, loss_focal,data):
    gray_batch = data["image"].cuda()
    aug_gray_batch = data["augmented_image"].cuda()
    anomaly_mask = data["anomaly_mask"].cuda()

    gray_rec = model(aug_gray_batch)
    joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)

    out_mask = model_seg(joined_in)
    out_mask_sm = torch.softmax(out_mask, dim=1)

    l2_loss = loss_l2(gray_rec,gray_batch)
    ssim_loss = loss_ssim(gray_rec, gray_batch)

    segment_loss = loss_focal(out_mask_sm, anomaly_mask)
    loss = l2_loss + ssim_loss + segment_loss

    return loss

def train_on_device(cfg):
    last_loss = 100

    dataset = MVTecDRAEMTrainDataset(cfg.data_dir + "leather/train/good/", cfg.anomaly_source_path, resize_shape=[cfg.image_size,cfg.image_size])
    train_dataset, val_dataset, threshold_dataset = split_data(dataset)

    train_dataset = DataLoader(train_dataset, batch_size=cfg.batch_size,shuffle=True)
    validate_dataset = DataLoader(val_dataset, batch_size=cfg.batch_size,shuffle=True)
    threshold_dataset = DataLoader(threshold_dataset, batch_size=cfg.batch_size,shuffle=True)

    model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    model.cuda()
    model.apply(weights_init)

    model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
    model_seg.cuda()
    model_seg.apply(weights_init)

    optimizer = torch.optim.Adam([
                {"params": model.parameters(), "lr": cfg.lr},
                {"params": model_seg.parameters(), "lr": cfg.lr}])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[cfg.n_epochs*0.8,cfg.n_epochs*0.9],gamma=0.2, last_epoch=-1)

    loss_l2 = torch.nn.modules.loss.MSELoss()
    loss_ssim = SSIM()
    loss_focal = FocalLoss()

    n_epochs=0
    for epoch in range(cfg.n_epochs):
        total_train_loss = 0
        total_validation_loss = 0
        n_epochs+=1
        
        for i_train, train_batched in enumerate(train_dataset):          
            train_loss = train_one_step(model,model_seg, loss_l2, loss_ssim, loss_focal, scheduler, optimizer,train_batched)
            total_train_loss += train_loss.data
        
        for i_validate, validate_batched in enumerate(validate_dataset):
            validation_loss =validate_one_step(model,model_seg, loss_l2, loss_ssim, loss_focal,validate_batched)
            total_validation_loss += validation_loss.data
        
        print('epoch [{}/{}], loss:{:.4f}, val_loss:{:.4f}'.format(epoch+1, cfg.n_epochs, total_train_loss, total_validation_loss))    
    
                #check Early stopping
        if total_validation_loss > last_loss:
            trigger_times += 1
            if trigger_times >= cfg.patience:
                print('Early stopping!\n Not improve during the last 20 epochs.')
                break
        else:
            trigger_times = 0
            #save model
            torch.save(model.state_dict(), os.path.join("model_join.pckl"))
            torch.save(model_seg.state_dict(), os.path.join("seg_model_join0.pckl"))

            last_loss = total_validation_loss

    threshold=get_threshold(threshold_dataset,model, model_seg,cfg)

    return threshold
        
def test_on_device(cfg):
    threshold = cfg.train_threshold
    model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    model.load_state_dict(torch.load("model.pckl"))
    model.cuda()
    model.eval()
    
    model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
    model_seg.load_state_dict(torch.load("seg_model.pckl"))
    model_seg.cuda()
    model_seg.eval()
    
    dataset = MVTecDRAEMTestDataset(cfg.data_dir + 'leather' + "/test", resize_shape=[cfg.image_size,cfg.image_size]) 
    test_loader, threshold_loader = split_data_testing(dataset)

    test_dataset = DataLoader(test_loader, batch_size=1,shuffle=True)
    threshold_dataset = DataLoader(threshold_loader, batch_size=1,shuffle=True)

    det_threshold, seg_threshold=optime_threshold(threshold_dataset,model,cfg)
    
    images, gts, labels, scores = [], [], [], []
    with torch.no_grad():
        for i, element in enumerate(test_dataset):
  
            gray_batch = element["image"].cuda()
            is_normal = element["has_anomaly"].detach().numpy()[0 ,0]
            true_mask = element["mask"]

            gray_rec = model(gray_batch)

            joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

            out_mask = model_seg(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)

            labels.append(is_normal)
            images.extend(gray_batch.detach().cpu().numpy())
            gts.extend(true_mask.cpu().numpy())
            scores.append(out_mask_sm[0 ,1 ,: ,:].detach().cpu().numpy())

    
    # gc.collect() 
    det_roc_auc,seg_roc_auc, seg_iou_training_threshold, seg_iou_optimized_threshold, f1score_training_threshold, f1score_optimized_threshold,pro_training_threshold, pro_segmentation_threshold = evaluate(gts,scores,labels,threshold,det_threshold,seg_threshold)

    print("detection auroc: ", det_roc_auc)
    print("detection f1 score training threshold: ", f1score_training_threshold)
    print("detection f1 score optimized threshold: ", f1score_optimized_threshold)
    print("segmentation auroc: ", seg_roc_auc)
    print("segmentation training threshold: ", seg_iou_training_threshold)
    print("segmentation optimized threshold: ", seg_iou_optimized_threshold)
  
    #cm is a function to convert gray images to viridis color map
    cm = plt.get_cmap('viridis')
    
    for image, mask, label, map in zip (images,gts,labels,scores):

        image, gt, super_mask, t1,t2,t3 = get_image(image,mask,map,threshold,det_threshold,seg_threshold)
        map=cm(super_mask)

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
cfg = Config().parse()

with torch.cuda.device(0):
    #threshold = train_on_device(cfg) 
    test_on_device(cfg,0.97)

with torch.cuda.device(0):
    if(cfg.phase=="train"): 
        threshold=train_on_device(cfg)
    elif(cfg.phase=="test"):
        test_on_device(cfg)
    else:
        print("Insert train or test in --phase argument")
