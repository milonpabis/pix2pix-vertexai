from torch.utils.tensorboard import SummaryWriter
import PIL
import torch
from torchvision.utils import save_image
import datetime as dt



class Logger:

    def __init__(self, log_dir: str, log_image_dir: str, checkpoint_dir: str):
        self.writer = SummaryWriter(log_dir)
        self.log_image_dir = log_image_dir
        self.checkpoint_dir = checkpoint_dir

    def log_scalar(self, tag: str, value: float, batch: int, epoch: int):
        self.writer.add_scalar(tag, value, (10000*epoch + batch)/10000)

    def log_image(self, tag: str, image_in: torch.Tensor, image_out: torch.Tensor, batch: int, epoch: int):
        img = torch.cat((image_in, image_out), 0)
        save_image(img, f"{self.log_image_dir}/{tag}_{epoch}_{batch}.png")

    def save_checkpoint(self, gen_state: dict, disc_state: dict, gen_opt_state: dict, disc_opt_state: dict, epoch: int):
        torch.save({
            'epoch': epoch,
            'gen_state_dict': gen_state,
            'disc_state_dict': disc_state,
            'gen_opt_state_dict': gen_opt_state,
            'disc_opt_state_dict': disc_opt_state
        }, f"{self.checkpoint_dir}/checkpoint_{epoch}_{dt.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}.pt")