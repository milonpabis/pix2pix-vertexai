import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

import google.cloud.storage as gcs
from google.oauth2 import service_account
import os
import datetime as dt

class Logger:

    def __init__(
            self, 
            log_dir: str, 
            log_image_dir: str, 
            checkpoint_dir: str, 
            validation_data_handler,
            credentials: service_account.Credentials = None,
            bucket_name: str = None
    ):
        try:
            if credentials:
                self.client = gcs.Client(credentials=credentials)
            else:
                self.client = gcs.Client()
                
            self.bucket = self.client.bucket(bucket_name=bucket_name)
        except Exception as e:
            print(e)
            self.bucket = None

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
            img = torch.cat((x, y, y_fake), 0) * 0.5 + 0.5
            file_name = f"{self.log_image_dir}/{tag}_{epoch}_{batch}.png"
            save_image(img, file_name)
            
            self._save_to_gcs(file_name)

        generator.train()

    def save_checkpoint(
            self, 
            gen_state: dict, 
            disc_state: dict, 
            gen_opt_state: dict, 
            disc_opt_state: dict, 
            epoch: int
        ):
        file_name = f"{self.checkpoint_dir}/checkpoint_{epoch}_{dt.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}.pt"

        torch.save(
            {
                'epoch': epoch,
                'gen_state_dict': gen_state,
                'disc_state_dict': disc_state,
                'gen_opt_state_dict': gen_opt_state,
                'disc_opt_state_dict': disc_opt_state
            },
            file_name
        )

        self._save_to_gcs(file_name)


    def _save_to_gcs(self, file_name: str):
        if self.bucket:
            blob = self.bucket.blob(file_name)
            with open(file_name, "rb") as f:
                blob.upload_from_file(f)

            os.remove(file_name)