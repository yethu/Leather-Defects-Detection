import argparse

class Config:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--phase", type=str, required=True, help="--train or --test")
        self.parser.add_argument("--data_dir", type=str, default=r"./datasets/neadvance/", help="Train directory folder path")
        self.parser.add_argument('--anomaly_source_path', action='store', type=str, default='./datasets/dtd/images/')
        self.parser.add_argument("--image_size", type=int, default=256, help="Size to reshape the image, the image will be square, height equal to width")
        self.parser.add_argument("--batch_size", type=int, default=1)
        self.parser.add_argument("--n_epochs", type=int, default=500)
        self.parser.add_argument("--lr", type=float, default=0.0001)
        self.parser.add_argument("--decay", type=float, default=1e-5)
        self.parser.add_argument("--patience", type=float, default=500)
        self.parser.add_argument("--train_threshold", type=float, default=0.5)
    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt

  