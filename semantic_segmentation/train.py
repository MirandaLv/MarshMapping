
import os
from os.path import dirname as up
from typing import Any, Dict, Type, cast
import pytorch_lightning as pl
from omegaconf import OmegaConf
# from trainer import MarshMapping

from MarshModel import MarshModel
import smp_metrics

import pandas as pd
from dataloader import GenMARSH, RandomRotation, Resize
import torch
from torch.utils.data import DataLoader
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import imgaug as ia
import imgaug.augmenters as iaa
import torchvision.transforms as transforms
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


torch.manual_seed(0)


if __name__ == "__main__":
    
    conf = OmegaConf.load("./config_smp.yaml")
    conf_dict = OmegaConf.to_object(conf.experiment)
    conf_dict = cast(Dict[Any, Dict[Any, Any]], conf_dict)

    # prepare data for training
    dl_kwargs = conf_dict["dataloader"]
    tmi_file = dl_kwargs["labelpath"]
    data_root = dl_kwargs["data_root_dir"]
    
#     transform_train = transforms.Compose([transforms.ToTensor(),
#                                     Resize(256),
#                                     transforms.RandomHorizontalFlip()])
    
    
    # load data
    dataset = GenMARSH(tmi_file, data_root, transform=None, normalization=dl_kwargs['normalization'], ndvi=dl_kwargs['ndvi'], ndwi=dl_kwargs['ndwi'], datasource=dl_kwargs['datasource'])

    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(0))

    trainloader = DataLoader(train_dataset, 
                    batch_size = dl_kwargs["batch_size"], 
                    shuffle = True,
                    num_workers = dl_kwargs["num_workers"],
                    pin_memory = False)

    testloader = DataLoader(test_dataset, 
                    batch_size = dl_kwargs["batch_size"], 
                    shuffle = False,
                    num_workers = dl_kwargs["num_workers"],
                    pin_memory = False)
    
    # load model parameters
    model_kwargs = conf_dict["module"]
    model = MarshModel(**model_kwargs)
    
    trainer = pl.Trainer(s
        max_epochs=100,
        accelerator="gpu",
        devices=[5]
    )

    trainer.fit(
        model, 
        train_dataloaders=trainloader, 
        val_dataloaders=testloader,
    )

    
# #     experiment_name = f"{model_kwargs['arch']}_{model_kwargs['encoder_name']}_0.0001"
# #     experiment_dir = os.path.join(dl_kwargs["out_dir"], experiment_name)
# #     logger = TensorBoardLogger(experiment_dir, name="models")

    
#     early_stop_callback = EarlyStopping(monitor="valid_dataset_iou", min_delta=0.00, patience=5, verbose=False, mode="max")
# #     checkpoint_callback = ModelCheckpoint(
# #             monitor='valid_dataset_iou', dirpath=experiment_dir, save_top_k=1, save_last=True, mode='max')
    
#     trainer = pl.Trainer(
#             callbacks=[early_stop_callback],
# #             logger=logger,
# #             default_root_dir=experiment_dir,
#             min_epochs=1,
#             max_epochs=100,
#             accelerator="gpu",
#             devices=[3])

#     trainer.fit(
#         model, 
#         train_dataloaders=trainloader, 
#         val_dataloaders=testloader,
#     )





##############################
# torchgeo trainer
# model = MarshMapping(**model_kwargs)

# # Instantiate trainer
# early_stop_callback = EarlyStopping(monitor="test_loss", min_delta=0.00, patience=10, verbose=False, mode="min")

# trainer = pl.Trainer(
#     gpus=1,
#     max_epochs=100,
#     callbacks=[early_stop_callback]
# )

# trainer.fit(
#     model, 
#     train_dataloaders=trainloader, 
#     val_dataloaders=testloader,
# )
##############################


