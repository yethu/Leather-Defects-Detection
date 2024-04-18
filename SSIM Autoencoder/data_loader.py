from torch.utils.data import Dataset
import torch
from skimage import io, transform, filters, color
import numpy as np
import glob


class TrainDataset(Dataset):
    def __init__(self,path,image_shape):
        self.path=path
        self.images = sorted(glob.glob(path+"/*/*.png"))
        self.image_shape=image_shape
    
    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path):
        image = io.imread(image_path)
        if(image.shape[2]==4):
            image= color.rgba2rgb(image)
        image = transform.resize(image, self.image_shape)
        image = filters.gaussian(image, sigma=0.4)

        image = np.transpose(image, (2, 0, 1))
        return image

    def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()

            img_path=self.images[idx]
            image = self.transform_image(img_path)
    
            sample = {'image': image}

            return sample


class TestDataset(Dataset):
    def __init__(self,image_paths,mask_paths,image_shape,mask_shape):
        self.image_path = image_paths
        self.ground_truths = mask_paths
        self.images = sorted(glob.glob(image_paths+"/*/*.png"))
        self.masks = sorted(glob.glob(mask_paths+"/*/*.png"))
        self.image_shape = image_shape
        self.mask_shape = mask_shape
    
    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path):
        image = io.imread(image_path)
        if(image.shape[2]==4):
            image= color.rgba2rgb(image)
        image = transform.resize(image, self.image_shape)
        image = filters.gaussian(image, sigma=0.4)

        image = np.transpose(image, (2, 0, 1))
        return image

    def transform_mask(self, mask_path):
        mask = io.imread(mask_path)
        mask = transform.resize(mask, self.mask_shape)

        threshold = filters.threshold_otsu(mask) 
        mask = mask > threshold
        mask = np.transpose(mask, (2, 0, 1))
        return mask

    def get_label(self,image_path):
        print('-----', image_path)
        defect_category = image_path.split('/')[-2]
        
        if defect_category == "good": return 0
        else: return 1

    def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()

            img_path=self.images[idx]
            image = self.transform_image(img_path)

            mask_path=self.masks[idx]
            mask = self.transform_mask(mask_path)
            
            label = self.get_label(img_path)
    
            sample = {'image': image, 'mask': mask ,'label': label}

            return sample