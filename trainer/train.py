import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import UNETGenerator, PatchGAN70x70
from datahandler import DataHandler
from logger import Logger
from utils import weights_init

import os
import datetime as dt
from tqdm import tqdm

# if __name__ == "__main__":

# parser = argparse.ArgumentParser()
# parser.add_argument("--epoch", type=int, default=0, help="starting_epoch")
# parser.add_argument("--num_epochs", type=int, default=EPOCHS, help="number of epochs")
# parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="size of the batches")
# parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="adam: learning rate")
# parser.add_argument("--b1", type=float, default=BETA1, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--b2", type=float, default=BETA2, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--train_path", type=str, default="data/building_facade/train", help="root path of the dataset")
# parser.add_argument("--val_path", type=str, default="data/building_facade/test", help="root path of the dataset")
# parser.add_argument("--checkpoint_interval", type=int, default=30, help="interval between model checkpoints")
# parser.add_argument("--checkpoint_path", type=str, default="checkpoints", help="path to save checkpoints")
# parser.add_argument("--log_path", type=str, default="run/pix2pix", help="path to save logs")
# parser.add_argument("--results_path", type=str, default="results", help="path to save results")
# parser.add_argument("--load_model_path", type=str, default="", help="path to load model from")
# parser.add_argument("--L1lambda", type=float, default=100, help="weight for L1 loss")
# parser.add_argument("--transform_type", type=str, default="augment", help="augment data")
# opt = parser.parse_args()
BUCKET_NAME = "pix2pix-training-artifacts-1171246"
CREDENTIALS_PATH = "sandbox-project-462110-215ba3072b18.json"
bucket_uri = f"gs://{BUCKET_NAME}"

class Opt:
    epoch = 0
    num_epochs = 200
    batch_size = 1
    lr = 2e-4
    b1 = 0.5
    b2 = 0.999
    train_path = "data/train"
    val_path = "data/val"
    checkpoint_interval = 50
    checkpoint_path = "checkpoints"
    log_path = "run/pix2pix"
    results_path = f"results/{dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}"
    load_model_path = ""
    L1lambda = 100
    transform_type = "augment"




def main():
    opt = Opt()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")

    if not os.path.exists(opt.log_path):
        os.makedirs(opt.log_path, exist_ok=True)

    if not os.path.exists(opt.results_path):
        os.makedirs(opt.results_path, exist_ok=True)

    if not os.path.exists(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path, exist_ok=True)


    generator = UNETGenerator(3).to(device)
    discriminator = PatchGAN70x70(3).to(device)

    weights_init(generator)
    weights_init(discriminator)

    gen_opt = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=1e-5)
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=1e-5)

    # Load checkpoints if available
    if len(opt.load_model_path):
        state_dict = torch.load(opt.load_model_path)
        generator.load_state_dict(state_dict["gen_state_dict"])
        gen_opt.load_state_dict(state_dict["gen_opt_state_dict"])
        discriminator.load_state_dict(state_dict["disc_state_dict"])
        disc_opt.load_state_dict(state_dict["disc_opt_state_dict"])
        start_epoch = torch.load(state_dict["epoch"])
        print(f"Resuming training from epoch {opt.epoch}")

    L1 = nn.L1Loss().to(device)
    BCE = nn.BCEWithLogitsLoss().to(device)

    datahandler = DataHandler(opt.train_path, target_side="left")
    dataloader = DataLoader(datahandler, batch_size=opt.batch_size, shuffle=True)

    datahandler_val = DataHandler(opt.val_path, target_side="left")
    dataloader_val = DataLoader(datahandler_val, batch_size=opt.batch_size, shuffle=True)

    logger = Logger(
        opt.log_path, 
        opt.results_path, 
        opt.checkpoint_path, 
        dataloader_val,
        credentials_path=CREDENTIALS_PATH,
        bucket_name=BUCKET_NAME)

    generator_scaler = torch.GradScaler()
    discriminator_scaler = torch.GradScaler()

    # autocast to float16
    # logging the results (losses, images)
    # saving the model (checkpoints)
    # option to resume training with checkpoints (also store optimizer states and epoch number)

    for epoch in range(opt.epoch+1, opt.num_epochs+1):    # epochs
        loop = tqdm(dataloader, leave=True)

        for idx, (x, y) in enumerate(loop): # batches
            x, y = x.to(device), y.to(device)


            with torch.autocast("cuda"): # training the discriminator
                y_fake = generator(x)
                d_real = discriminator(x, y) # showing the discriminator the real image and real output
                d_fake = discriminator(x, y_fake.detach()) # showing the discriminator the real image and fake output

                d_real_loss = BCE(d_real, torch.ones_like(d_real))
                d_fake_loss = BCE(d_fake, torch.zeros_like(d_fake))

                d_loss = (d_real_loss + d_fake_loss) / 2


            disc_opt.zero_grad()
            discriminator_scaler.scale(d_loss).backward() #loss
            discriminator_scaler.step(disc_opt)
            discriminator_scaler.update()


            with torch.autocast("cuda"): # training the generator
                d_fake = discriminator(x, y_fake)
                g_fake_loss = BCE(d_fake, torch.ones_like(d_fake))
                l1 = L1(y_fake, y) * opt.L1lambda

                g_loss = g_fake_loss + l1

            gen_opt.zero_grad()
            generator_scaler.scale(g_loss).backward()
            generator_scaler.step(gen_opt)
            generator_scaler.update()

            if idx % 500 == 0 and idx != 0:
                logger.log_scalar("d_loss", d_loss.item(), idx, epoch)
                logger.log_scalar("g_loss", g_loss.item(), idx, epoch)
                logger.log_scalar("l1_loss", l1.item(), idx, epoch)
                logger.log_scalar("d_fake_loss", d_fake_loss.item(), idx, epoch)
                logger.log_scalar("d_real_loss", d_real_loss.item(), idx, epoch)
                logger.log_scalar("g_fake_loss", g_fake_loss.item(), idx, epoch)
                logger.log_image("result", generator, idx, epoch)

        # loop.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())

        if epoch % opt.checkpoint_interval == 0 and epoch != 0:
            logger.save_checkpoint(generator.state_dict(), discriminator.state_dict(), gen_opt.state_dict(), disc_opt.state_dict(), epoch)

if __name__ == "__main__":

    main()
    # download_dataset(BUCKET_NAME, "data_maps/val", "data/val")