import argparse

class Config:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--phase", type=str, required=True, help="--train or --test")
        self.parser.add_argument("--train_data_dir", type=str, default=r"C:\Users\jpmrs\OneDrive\Desktop\Dissertação\code\Data\join\leather\train", help="Train directory folder path")
        self.parser.add_argument("--test_data_dir", type=str, default=r"C:\Users\jpmrs\OneDrive\Desktop\Dissertação\code\Data\mvtec\leather\test",  help="Test directory folder path")
        self.parser.add_argument("--mask_data_dir", type=str, default=r"C:\Users\jpmrs\OneDrive\Desktop\Dissertação\code\data\mvtec\leather\ground_truth")
        self.parser.add_argument("--image_size", type=int, default=256, help="Size to reshape the image, the image will be square, height equal to width")
        self.parser.add_argument("--batch_size", type=int, default=64)
        self.parser.add_argument("--n_epochs", type=int, default=50)
        self.parser.add_argument("--lr", type=float, default=0.001)
        self.parser.add_argument("--decay", type=float, default=1e-5)
        self.parser.add_argument("--patience", type=float, default=500)

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt

  