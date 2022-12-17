import wandb
from config import Config
import torch
from torch.utils.data import DataLoader
from data_loader import TrainDataset, TestDataset
from utils import *
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from scipy.ndimage import gaussian_filter


def get_threshold(dataset,encoder,decoder,bn,cfg):
    scores = []
    for batch in dataset:
        for image in batch["image"]:
            image = image.unsqueeze(0) 
            image = image.cuda()
            image = image.float().cuda()
        
            with torch.set_grad_enabled(False):
                inputs = encoder(image)
                outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, image.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            anomaly_map = anomaly_map.reshape(cfg.image_size,cfg.image_size) 
            scores.append(anomaly_map)
         
    scores = np.array(scores)

    return float(np.percentile(scores, [99]))

def train_one_step(encoder,decoder,bn,criterion,optimizer,data):
    image_batch = data["image"].cuda()
    image_batch = image_batch.float().cuda()
    
    bn.train()
    decoder.train()

    optimizer.zero_grad()
    with torch.set_grad_enabled(True): 
        # ===================forward=====================
        inputs = encoder(image_batch)

        outputs = decoder(bn(inputs))#bn(inputs))
        loss = loss_function(inputs, outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return loss
    
def validate_one_step(encoder,decoder,bn,criterion,data):
    image_batch = data["image"].cuda()
    image_batch = image_batch.float().cuda()
    
    bn.eval()
    decoder.eval()

   
    with torch.set_grad_enabled(True): 
        # ===================forward=====================
        inputs = encoder(image_batch)

        outputs = decoder(bn(inputs))#bn(inputs))
        loss = loss_function(inputs, outputs)
       
    return loss

def train_on_device(cfg):
        
    last_loss=100
    image_shape = (cfg.image_size, cfg.image_size, 3)

    #load datasets
    dataset = TrainDataset(path=cfg.train_data_dir, image_shape=image_shape) 
    train_dataset, val_dataset, threshold_dataset = split_data(dataset)
    
    train_dataset = DataLoader(train_dataset, batch_size=cfg.batch_size,shuffle=True)
    validate_dataset = DataLoader(val_dataset, batch_size=cfg.batch_size,shuffle=True)
    threshold_dataset = DataLoader(threshold_dataset, batch_size=cfg.batch_size,shuffle=True)
    
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.cuda()
    bn = bn.cuda()
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.cuda()

    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=cfg.lr, betas=(0.5,0.999))
    criterion = loss_function
    n_epochs=0

    for epoch in range(cfg.n_epochs):   
        total_train_loss = 0
        total_validation_loss = 0
        n_epochs+=1 

        bn.train()
        decoder.train()
       
        for i_train, train_batched in enumerate(train_dataset):          
            train_loss = train_one_step(encoder,decoder,bn,criterion,optimizer,train_batched)
            total_train_loss += train_loss.data
        
        for i_validate, validate_batched in enumerate(validate_dataset):
            validation_loss = validate_one_step(encoder,decoder,bn,criterion,validate_batched)
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
            state ={'bn': bn.state_dict(),'decoder': decoder.state_dict()}
            torch.save(state, "model.pckl")
            
            
        
            last_loss = total_validation_loss

    threshold=get_threshold(threshold_dataset,encoder,decoder,bn,cfg)    
    print("Train Threshold", threshold)
    return threshold

def test_on_device(cfg):
    threshold = cfg.train_threshold
    image_shape = (cfg.image_size, cfg.image_size, 3)
    mask_shape =  (cfg.image_size, cfg.image_size, 1)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.cuda()
    bn = bn.cuda()
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.cuda()
    
    ckp = torch.load("model.pckl")
    
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)

    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])

    dataset = TestDataset(cfg.test_data_dir, cfg.mask_data_dir, image_shape, mask_shape) 
    test_loader, threshold_loader = split_data_testing(dataset)

    test_dataset = DataLoader(test_loader, batch_size=1,shuffle=True)
    threshold_dataset = DataLoader(threshold_loader, batch_size=1,shuffle=True)

    det_threshold, seg_threshold=optime_threshold(threshold_dataset,encoder,bn,decoder,cfg)
    
    images, gts, labels, scores = [], [], [], []

    with torch.no_grad():
        for element in test_dataset:
  
            img = element['image'].cuda()
            img = img.float().cuda()
            gt = element['mask']
            gt = gt.float()
            label = element['label']

            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            anomaly_map = anomaly_map.reshape(cfg.image_size,cfg.image_size) 

            labels.append(label)
            images.extend(img.cpu().numpy())
            gts.extend(gt.cpu().numpy())
            scores.append(anomaly_map)      
    
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
        map=cm(map)

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
