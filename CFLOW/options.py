import argparse

class Options:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--phase", type=str, required=True, help="--train or --test")
        self.parser.add_argument("--train_data_dir", type=str, default=None, help="Train directory folder path")
        self.parser.add_argument("--test_data_dir", type=str, default=None,  help="Test directory folder path")
        self.parser.add_argument("--mask_data_dir", type=str, default=None)
        self.parser.add_argument("--image_size", type=int, default=256, help="Size to reshape the image, the image will be square, height equal to width")
        self.parser.add_argument("--batch_size", type=int, default=64)
        self.parser.add_argument('-enc', '--enc-arch', default='resnet18', type=str, metavar='A',help='feature extractor: wide_resnet50_2/resnet18/mobilenet_v3_large (default: wide_resnet50_2)')
        self.parser.add_argument('-dec', '--dec-arch', default='freia-cflow', type=str, metavar='A', help='normalizing flow model (default: freia-cflow)')
        self.parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',help='learning rate (default: 2e-4)')
        self.parser.add_argument('--sub-epochs', type=int, default=8, metavar='N',help='number of sub epochs to train (default: 8)')
        self.parser.add_argument('--meta-epochs', type=int, default=63, metavar='N',help='number of meta epochs to train (default: 25)')
        self.parser.add_argument('-pl', '--pool-layers', default=3, type=int, metavar='L',help='number of layers used in NF model (default: 3)')
        self.parser.add_argument('-cb', '--coupling-blocks', default=8, type=int, metavar='L',help='number of layers used in NF model (default: 8)')
        self.parser.add_argument("--patience", type=float, default=20)
        self.parser.add_argument("--train_threshold", type=float, default=0.5)
        
    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt

  
  