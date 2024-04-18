from cv2 import threshold
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import sklearn.metrics as metrics
from segmentation_models.metrics import IOUScore 
from skimage.segmentation import mark_boundaries, clear_border
from skimage.measure import label, regionprops
from skimage import morphology

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

def optime_threshold(loader,model,cfg):
    true_masks, residuals, labels = [], [], []
    for i_batch, sample_batched in enumerate(loader):

        image_batch = sample_batched["image"].cuda()
        image_batch = image_batch.float().cuda()
        mask_batch = sample_batched["mask"].cuda()
        mask_batch = mask_batch.float().cuda()
        label_batch = sample_batched["label"]

        preds,residual_maps = get_residual_map(image_batch,model,cfg)   
        residual_maps = np.array(residual_maps) # Creating a tensor from a list of numpy.ndarrays is extremely slow. So i convert the list of numpy.ndarrays to a numpy.ndarrays
        
        true_masks.extend(mask_batch.cpu().data.numpy())
        labels.extend(label_batch.numpy())
        residuals.extend(residual_maps)

    score_label = np.max(residuals, axis=(1, 2))
    gt_label = np.asarray(labels, dtype=bool)
    precision, recall, thresholds = metrics.precision_recall_curve(gt_label,score_label)

    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    det_threshold = thresholds[np.argmax(f1)]

    true_masks = np.squeeze(np.asarray(true_masks, dtype=bool))
    residuals = np.squeeze(np.asarray(residuals))
    precision, recall, thresholds = metrics.precision_recall_curve(true_masks.flatten(), residuals.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    seg_threshold = thresholds[np.argmax(f1)]

    return det_threshold, seg_threshold

def get_threshold(dataset,model,cfg):
    total_rec= []
    for i_batch, batch in enumerate(dataset): 
            image_batch = batch["image"].cuda()
            image_batch = image_batch.float().cuda()
            results,residual_maps = get_residual_map(image_batch,model,cfg)
            total_rec.extend(residual_maps)
         
    total_rec = np.array(total_rec)

    return float(np.percentile(total_rec, [99]))

def get_residual_map(batch, model,cfg):
    residual_maps=[]
    batch=batch.float().cuda()
    results =  model(batch)

    for image,result in zip(batch,results):
        image=image.permute(1,2,0)
        image = image.cpu().detach().numpy()
        result=result.permute(1,2,0)
        result = result.cpu().detach().numpy()
        
        residual_map = ssim(image, result, win_size=11, full=True, channel_axis=2)[1]
        residual_map = 1 - np.mean(residual_map, axis=2)
        
        residual_maps.append(residual_map)

    return results, residual_maps

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
    binary_score_maps = np.zeros_like(super_mask, dtype=bool)

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
    gt_label = np.asarray(labels, dtype=bool)
    gt_label[0] = False
    det_roc_auc = metrics.roc_auc_score(gt_label, score_label)

    gt_mask = np.squeeze(np.asarray(y_trues, dtype=bool), axis=1)
    seg_roc_auc = metrics.roc_auc_score(gt_mask.flatten(), residuals.flatten())

    #Iou training threshold and segmentation optimized threshold
    iou=IOUScore()

    seg_iou_training_threshold = iou(gt_mask, predicted_mask)
    seg_iou_optimized_threshold = iou(gt_mask, predicted_mask_segmentation)
  
    #detection f1 score and precision
    labels_training_threshold = [predicted_mask[i].max() for i in range(len(predicted_mask)) ]
    labels_detection_threshold = [predicted_mask_detection[i].max() for i in range(len(predicted_mask_detection)) ]
    
    f1score_training_threshold=metrics.f1_score(gt_label,labels_training_threshold)
    f1score_detection_threshold=metrics.f1_score(gt_label,labels_detection_threshold)

    #detection f1 score and precision
    labels_training_threshold = [predicted_mask[i].max() for i in range(len(predicted_mask)) ]
    labels_detection_threshold = [predicted_mask_detection[i].max() for i in range(len(predicted_mask_detection)) ]
    
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



    return image, gt, super_mask, t1,t2,t3

