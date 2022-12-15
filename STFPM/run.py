import wandb
from config import Config
import torch
from torch.utils.data import DataLoader
from data_loader import TrainDataset, TestDataset
from resnet_backbone import modified_resnet18
from utils import *

def get_threshold(dataset,model_s,model_t,cfg):
    scores = []
    for batch in dataset:
        for image in batch["image"]:
            image = image.unsqueeze(0) 
            image = image.cuda()
            image = image.float().cuda()
        
            with torch.set_grad_enabled(False):
                features_t = model_t(image)
                features_s = model_s(image)

            score = cal_anomaly_maps(features_s, features_t, cfg.crop_size)
            scores.append(score)
         
    scores = np.array(scores)

    return float(np.percentile(scores, [99]))

def train_one_step(model_s,model_t,criterion, optimizer,data):
    image_batch = data["image"].cuda()
    image_batch = image_batch.float().cuda()
    
    model_t.eval()
    model_s.train()

    optimizer.zero_grad()
    with torch.set_grad_enabled(True): 
        # ===================forward=====================
        features_t = model_t(image_batch)
        features_s = model_s(image_batch)
        # ===================backward====================
        
        loss = criterion(features_s, features_t)
        loss.backward()
        optimizer.step()
        
    return loss

def validate_one_step(model_s,model_t,criterion,data):
    image_batch = data["image"].cuda()
    image_batch = image_batch.float().cuda()
    
    model_t.eval()
    model_s.train()

    with torch.set_grad_enabled(True): 
        # ===================forward=====================
        features_t = model_t(image_batch)
        features_s = model_s(image_batch)
        # ===================backward====================
        
        loss = criterion(features_s, features_t)
        #loss.backward()
        
    return loss

def train_on_device(cfg):
    last_loss=100
    image_shape = (cfg.image_size, cfg.image_size, 3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #load datasets
    dataset = TrainDataset(path=cfg.train_data_dir, image_shape=image_shape) 
    train_dataset, val_dataset, threshold_dataset = split_data(dataset)
    
    train_dataset = DataLoader(train_dataset, batch_size=cfg.batch_size,shuffle=True)
    validate_dataset = DataLoader(val_dataset, batch_size=cfg.batch_size,shuffle=True)
    threshold_dataset = DataLoader(threshold_dataset, batch_size=cfg.batch_size,shuffle=True)

    model_t = modified_resnet18().to(device)
    model_s = modified_resnet18(pretrained=False).to(device)
    
    for param in model_t.parameters():
        param.requires_grad = False
    
    model_t.eval()

    criterion=cal_loss
    optimizer = torch.optim.SGD(model_s.parameters(), lr=0.4, momentum=0.9, weight_decay=0.0001)
    n_epochs=0

    for epoch in range(cfg.n_epochs):
        total_train_loss = 0
        total_validation_loss = 0
        n_epochs+=1
        
        for i_train, train_batched in enumerate(train_dataset):
            train_loss = train_one_step(model_s,model_t,criterion,optimizer,train_batched)
            total_train_loss += train_loss.data
        
        for i_validate, validate_batched in enumerate(validate_dataset):
            validation_loss = validate_one_step(model_s,model_t,criterion,validate_batched)
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
            state = {'model': model_s.state_dict()}
            torch.save(state, "model.pckl")
            
            last_loss = total_validation_loss

    threshold=get_threshold(threshold_dataset,model_s,model_t,cfg)

    return threshold
 
def test_on_device(cfg):
    threshold = cfg.train_threshold
    image_shape = (cfg.image_size, cfg.image_size, 3)
    mask_shape =  (cfg.image_size, cfg.image_size, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TestDataset(cfg.test_data_dir,cfg.mask_data_dir, image_shape=image_shape, mask_shape=mask_shape) 

    test_loader, threshold_loader = split_data_testing(dataset)

    test_dataset = DataLoader(test_loader, batch_size=cfg.batch_size,shuffle=True)
    threshold_dataset = DataLoader(threshold_loader, batch_size=cfg.batch_size,shuffle=True)

    model_t = modified_resnet18().to(device)
    model_s = modified_resnet18(pretrained=False).to(device)

    for param in model_t.parameters():
        param.requires_grad = False
  
    checkpoint = torch.load('model.pckl')
    model_s.load_state_dict(checkpoint['model'])
    model_t.eval()
      
    det_threshold, seg_threshold=optime_threshold(threshold_dataset,model_t,model_s,cfg)

    images, gts, labels, scores = [], [], [], []


    for element in test_dataset:
        
        img = element['image'].cuda()
        img = img.float().cuda()
        gt = element['mask']
        gt = gt.float()
        label = element['label']

        images.extend(img.squeeze().cpu().numpy())
        labels.extend(label.cpu().numpy())
        gts.extend(gt.squeeze().cpu().numpy())

        with torch.set_grad_enabled(False):
            features_t = model_t(img)
            features_s = model_s(img)

            score = cal_anomaly_maps(features_s, features_t, cfg.crop_size)
        scores.extend(score)
    
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
cfg = Config().parse()
cfg.crop_size = (cfg.image_size,cfg.image_size)  # HxW forma

with torch.cuda.device(0):
    if(cfg.phase=="train"): 
        threshold=train_on_device(cfg)
    elif(cfg.phase=="test"):
        test_on_device(cfg)
    else:
        print("Insert train or test in --phase argument")