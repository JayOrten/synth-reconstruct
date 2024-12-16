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

from models import Classifier
from dataset import AudioDS

def train(config): 
    print("loading dataset")
    dataset = AudioDS(config["presets_csv_path"], config["spectrograms_path"])
    pl.seed_everything(42)
    num_train = int(0.9 * len(dataset))
    num_val = len(dataset) - num_train
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [num_train, num_val])
    batch_size = config["batch_size"]

    # We define a set of data loaders that we can use for various purposes later.

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    print('creating trainer')
    trainer = pl.Trainer(
        default_root_dir=os.path.join(config["checkpoint_path"], "classifier_%i" % latent_dim),
        accelerator="auto",
        devices=1,
        max_epochs=config["max_epochs"],
        callbacks=[
            ModelCheckpoint(save_weights_only=True),
            LearningRateMonitor("epoch"),
        ],
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    print("creating model")
    model = Classifier(latent_dim=config["latent_dim"], num_classes=config["num_classes"], encoder_model_checkpoint=config["encoder_model_checkpoint"])
    print("training model")
    trainer.fit(model, train_loader, val_loader)
    # Test best model on validation and test set
    print("testing model")
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    # test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    # result = {"test": test_result, "val": val_result}
    result = {"val": val_result}


def main():
    # get parameters from config file, which is a yaml file
    with open(config, "r") as file:
        config = yaml.safe_load(file)

    train(config)

if __name__ == "__main__":
    main()
    