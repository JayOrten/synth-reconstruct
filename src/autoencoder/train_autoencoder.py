import os
import yaml
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
import sys

from models import Autoencoder
from dataset import AudioDS

class GenerateCallback(Callback):
    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs  # Images to reconstruct during training
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(input_imgs)
                pl_module.train()

            # print('input_imgs: ', input_imgs)
            # print('reconst_imgs: ', reconst_imgs)
            # Plot and add to tensorboard
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, value_range=(0, 1))
            # trainer.logger.experiment.image("reconstruction", grid, global_step=trainer.global_step)
            # save image to disk
            save_path = os.path.join(trainer.logger.log_dir, f"reconstruction_{trainer.current_epoch}.png")
            torchvision.utils.save_image(grid, save_path, nrow=2)

def get_train_images(num, train_dataset):
    return torch.stack([train_dataset[i][0] for i in range(num)], dim=0)

def train(config):
    print("loading dataset")
    dataset = AudioDS(config["presets_csv_path"], config["data_path"])
    pl.seed_everything(42)
    num_train = int(0.9 * len(dataset))
    num_val = len(dataset) - num_train
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [num_train, num_val])
    batch_size = config["batch_size"]

    # We define a set of data loaders that we can use for various purposes later.

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    # Create a PyTorch Lightning trainer with the generation callback
    print("creating trainer")
    trainer = pl.Trainer(
        default_root_dir=os.path.join(config["checkpoint_path"], "final_model_dim_%i" % config["latent_dim"]),
        accelerator="auto",
        devices=1,
        max_epochs=config["max_epochs"],
        callbacks=[
            ModelCheckpoint(save_weights_only=True),
            GenerateCallback(get_train_images(8, train_dataset), every_n_epochs=1),
            LearningRateMonitor("epoch"),
        ],
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    # pretrained_filename = os.path.join(CHECKPOINT_PATH, "cifar10_%i.ckpt" % latent_dim)
    # if os.path.isfile(pretrained_filename):
    #     print("Found pretrained model, loading...")
    #     model = Autoencoder.load_from_checkpoint(pretrained_filename)
    # else:
    print("creating model")
    model = Autoencoder(base_channel_size=32, latent_dim=config["latent_dim"])
    print("training model")
    trainer.fit(model, train_loader, val_loader)
    # Test best model on validation and test set
    print("testing model")
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    # test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    # result = {"test": test_result, "val": val_result}
    result = {"val": val_result}

def main():
    args = sys.argv
    config_path = args[1]
    # get parameters from config file, which is a yaml file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    train(config)

if __name__ == "__main__":
    main()