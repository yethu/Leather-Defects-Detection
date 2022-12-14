import torch
from torch.utils.data import DataLoader
import numpy as np
from config import Config
from data_loader import TrainDataset, TestDataset
from models import Autoencoder
from utils import *
from loss import SSIM
import time
import os


def train_one_step(model,criterion, optimizer,data):
    image_batch = data["image"].cuda()
    image_batch = image_batch.float().cuda()
    
    model.train()
    # ===================forward=====================
    output = model(image_batch)
    loss = criterion(image_batch,output)
    
    # ===================backward====================
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

def validate_one_step(model,criterion,data):
    image_batch = data["image"].cuda()
    image_batch = image_batch.float().cuda()

    model.eval()
    # ===================forward=====================
    output = model(image_batch)
    loss = criterion(image_batch,output)
    
    return loss

def train_on_device(cfg):   
    last_loss = 100
    n_channels=3
    image_shape = (cfg.image_size, cfg.image_size, n_channels)
    
    #load datasets
    dataset = TrainDataset(path=cfg.train_data_dir, image_shape=image_shape) 
    train_dataset, val_dataset, threshold_dataset = split_data(dataset)
    
    train_dataset = DataLoader(train_dataset, batch_size=cfg.batch_size,shuffle=True)
    validate_dataset = DataLoader(val_dataset, batch_size=cfg.batch_size,shuffle=True)
    threshold_dataset = DataLoader(threshold_dataset, batch_size=cfg.batch_size,shuffle=True)

    #define model, loss function and optimizer
    model = Autoencoder(n_channels).cuda()
     
    criterion = SSIM(in_channels=n_channels)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,weight_decay=cfg.decay)

    #train model and check early stopping
    n_epochs=0
    for epoch in range(cfg.n_epochs):
        total_train_loss = 0
        total_validation_loss = 0
        n_epochs+=1
        for i_train, train_batched in enumerate(train_dataset):          
            train_loss = train_one_step(model,criterion, optimizer,train_batched)
            total_train_loss += train_loss.data
        
        for i_validate, validate_batched in enumerate(validate_dataset):
            validation_loss = validate_one_step(model,criterion,validate_batched)
            total_validation_loss += validation_loss.data
        
        #check Early stopping
        if total_validation_loss > last_loss:
            trigger_times += 1
            if trigger_times >= cfg.patience:
                print('Early stopping!\n Not improve during the last 20 epochs.')
                break
        else:
            trigger_times = 0
            #save model
            torch.save(model.state_dict(), "model_ssim.pckl")
        
            last_loss = total_validation_loss
            print('epoch [{}/{}], loss:{:.4f}, val_loss:{:.4f}'.format(epoch+1, cfg.n_epochs, total_train_loss, total_validation_loss))   
    
    threshold=get_threshold(threshold_dataset,model,cfg)

    return model, threshold

def test_on_device(cfg, model, threshold):
    image_shape = (cfg.image_size, cfg.image_size,3)
    mask_shape =  (cfg.image_size, cfg.image_size, 1)

    #load datasets
    dataset = TestDataset(cfg.test_data_dir,cfg.mask_data_dir, image_shape=image_shape, mask_shape=mask_shape) 

    test_loader, threshold_loader = split_data_testing(dataset)

    test_dataset = DataLoader(test_loader, batch_size=cfg.batch_size,shuffle=True)
    threshold_dataset = DataLoader(threshold_loader, batch_size=cfg.batch_size,shuffle=True)

    #get optime threshold
    det_threshold, seg_threshold=optime_threshold(threshold_dataset,model,cfg)

    #evaluate model
    images, true_masks, predictions, residuals, labels = [], [], [], [], []
    
    for i_batch, sample_batched in enumerate(test_dataset):

        image_batch = sample_batched["image"].cuda()
        image_batch = image_batch.float().cuda()
        mask_batch = sample_batched["mask"].cuda()
        mask_batch = mask_batch.float().cuda()
        label_batch = sample_batched["label"]

        preds,residual_maps = get_residual_map(image_batch,model,cfg)   
        residual_maps = np.array(residual_maps) # Creating a tensor from a list of numpy.ndarrays is extremely slow. So i convert the list of numpy.ndarrays to a numpy.ndarrays
        
        images.extend(image_batch.cpu().data.numpy())
        true_masks.extend(mask_batch.cpu().data.numpy())
        labels.extend(label_batch.numpy())
        predictions.extend(preds.cpu().data.numpy())
        residuals.extend(residual_maps)

    det_roc_auc,seg_roc_auc, seg_iou_training_threshold, seg_iou_optimized_threshold, f1score_training_threshold, f1score_optimized_threshold,pro_training_threshold, pro_segmentation_threshold = evaluate(true_masks,residuals,labels,threshold,det_threshold,seg_threshold)

    print("detection auroc: ", det_roc_auc)
    print("detection f1 score training threshold: ", f1score_training_threshold)
    print("detection f1 score optimized threshold: ", f1score_optimized_threshold)
    print("segmentation auroc: ", seg_roc_auc)
    print("segmentation training threshold: ", seg_iou_training_threshold)
    print("segmentation optimized threshold: ", seg_iou_optimized_threshold)

    #cm is a function to convert gray images to viridis color map
    cm = plt.get_cmap('viridis')

    for image, mask, label, map in zip (images,true_masks,labels,residuals):

        image, gt, super_mask, true_boundary, t1,t2,t3 = get_image(image,mask,map,threshold,det_threshold,seg_threshold)
        super_mask=cm(super_mask)

        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 7)
        ax1.plot(image)
        ax2.plot(gt)
        ax3.plot(super_mask)
        ax4.plot(true_boundary)
        ax5.plot(t1)
        ax6.plot(t2)
        ax6.plot(t3)

# parse argument variables
cfg = Config().parse()

with torch.cuda.device(0):
    if(cfg.phase=="train"): 
        model, threshold = train_on_device(cfg) 
    elif(cfg.phase=="test"):
        model = Autoencoder(n_channels=3).cuda()
        model.load_state_dict(torch.load("model_ssim.pckl"))
        model.eval()
        test_on_device(cfg, model,0.43)
    else:
        print("Insert train or test in --phase argument")

