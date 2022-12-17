# Leather-Defects-Detection
This repository presents 5 distinct techniques based on Novelty Detection, implemented using Pytorch:

1. SSIM Autoencoder (https://arxiv.org/abs/1807.02011)
2. CFLOW (https://arxiv.org/abs/2107.12571)
3. STFPM (https://arxiv.org/abs/2103.04257)
4. Reverse Distillation (https://arxiv.org/abs/2201.10703)
5. DRAEM (https://arxiv.org/abs/2108.07610)

For this problem, the MVTec AD dataset (https://www.mvtec.com/company/research/datasets/mvtec-ad) is used in benchmarks. The dataset consist of three folders:

* train, which contains the (defect-free) training images
* test, which contains the test images
* ground_truth, which contains the pixel-precise annotations of anomalous regions

You can try this techniques with others datasets, you just need to follow the same dataset structure.
In this work, 3 distinct threshols are used. One is estimated using a split of the training data and the others are estimated using a split of the test dataset.

To execute this techniques, it is necessary to run the following commands inside the techniques folder:
This command only have the most important arguments, you can find the others in config files.

1. SSIM Autoencoder

    python run.py --phase "train" --train_data_dir "C:\path\train" --image_size 256 --batch_size=64 --n_epochs 50 

    python run.py --phase "test" --test_data_dir "C:\path\test" --ground-truth "C:\path\ground_truth" --image_size 256 --train_threshold 0.5


2. CFLOW 

    python run.py --phase "train" --train_data_dir "C:\path\train" --image_size 256 --batch_size=64 --sub-epochs 1 --meta-epochs 5

    python run.py --phase "test" --test_data_dir "C:\path\test" --ground-truth "C:\path\ground_truth" --image_size 256 --train_threshold 0.5

3. STFPM 

    python run.py --phase "train" --train_data_dir "C:\path\train" --image_size 256 --batch_size=64 --n_epochs 50 

    python run.py --phase "test" --test_data_dir "C:\path\test" --ground-truth "C:\path\ground_truth" --image_size 256 --train_threshold 0.5

4. Reverse Distillation 

    python run.py --phase "train" --train_data_dir "C:\path\train" --image_size 256 --batch_size=64 --n_epochs 50 

    python run.py --phase "test" --test_data_dir "C:\path\test" --ground-truth "C:\path\ground_truth" --image_size 256 --train_threshold 0.5
    
5. DRAEM 
    See original repo (https://github.com/VitjanZ/DRAEM/blob/main/train_DRAEM.py) to download textures dataset
    python run.py --phase "train" --data_dir "C:\path" ---image_size 256 --batch_size=64 --n_epochs 50 

    python run.py --phase "test" --data_dir "C:\path\" --image_size 256 --train_threshold 0.5

