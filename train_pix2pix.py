import os
import argparse
from models.discriminator import PatchGAN70x70
from models.generator import UNETGenerator
from models.settings import *
from models.__init__ import *


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=0, help="starting_epoch")
parser.add_argument("--num_epochs", type=int, default=EPOCHS, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="size of the batches")
parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=BETA1, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=BETA2, help="adam: decay of first order momentum of gradient")
parser.add_argument("--dataset_path", type=str, default="data/building_facade", help="root path of the dataset")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between model checkpoints")
opt = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda")



if __name__ == "__main__":
    ...
