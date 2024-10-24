from torch.utils.tensorboard import SummaryWriter
import PIL
import torch
from torchvision.utils import save_image
import datetime as dt
import random as rd



class Logger:

    def __init__(self, log_dir: str, log_image_dir: str, checkpoint_dir: str, validation_data_handler):
        self.writer = SummaryWriter(log_dir)
        self.log_image_dir = log_image_dir
        self.checkpoint_dir = checkpoint_dir
        self.val_handler = validation_data_handler

    def log_scalar(self, tag: str, value: float, batch: int, epoch: int):
        self.writer.add_scalar(tag, value, (10000*epoch + batch)/10000)

    def log_image(self, tag: str, generator, batch: int, epoch: int):
        x, y = next(iter(self.val_handler))
        x, y = x.to("cuda"), y.to("cuda")
        generator.eval()
        with torch.no_grad():
            with torch.autocast("cuda"):
                y_fake = generator(x)
            img = torch.cat((x, y, y_fake), 0)
            save_image(img, f"{self.log_image_dir}/{tag}_{epoch}_{batch}.png")
        generator.train()

    def save_checkpoint(self, gen_state: dict, disc_state: dict, gen_opt_state: dict, disc_opt_state: dict, epoch: int):
        torch.save({
            'epoch': epoch,
            'gen_state_dict': gen_state,
            'disc_state_dict': disc_state,
            'gen_opt_state_dict': gen_opt_state,
            'disc_opt_state_dict': disc_opt_state
        }, f"{self.checkpoint_dir}/checkpoint_{epoch}_{dt.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}.pt")