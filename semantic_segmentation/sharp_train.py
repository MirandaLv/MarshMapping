
import os
from os.path import dirname as up
from typing import Any, Dict, Type, cast
import pytorch_lightning as pl
from omegaconf import OmegaConf

# from MarshModel import MarshModel
from sharp_trainer import SemanticSegmentationTask
import smp_metrics

import pandas as pd
from sharp_dataloader import GenMARSH, RandomRotation, Resize
import torch
from torch.utils.data import DataLoader
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import imgaug as ia
import imgaug.augmenters as iaa
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import itertools 


torch.manual_seed(0)


if __name__ == "__main__":
    
    conf = OmegaConf.load("./test_multiclass.yaml")
    conf_dict = OmegaConf.to_object(conf.experiment)
    conf_dict = cast(Dict[Any, Dict[Any, Any]], conf_dict)

    # prepare data for training
    dl_kwargs = conf_dict["dataloader"]
    label_file = dl_kwargs["labelpath"]

    # load data
    dataset = GenMARSH(label_file, transform=None, normalization=dl_kwargs['normalization'], ndvi=dl_kwargs['ndvi'], ndwi=dl_kwargs['ndwi'], datasource=dl_kwargs['datasource'])

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

    # Setup model
#     model_kwargs = conf_dict['module']
    
#     experiment_folder = "{}_{}_{}_{}_{}_{}".format(model_kwargs['segmentation_model'], model_kwargs['encoder_name'], model_kwargs['learning_rate'], model_kwargs['in_channels'], model_kwargs['loss'], dl_kwargs['tracker_val'])
    
#     experiment_dir = os.path.join(dl_kwargs["out_dir"], experiment_folder)
#     logger = TensorBoardLogger(experiment_dir, name="models")
    
#     checkpoint_callback = ModelCheckpoint(
#         monitor=dl_kwargs['tracker_val'], dirpath=experiment_dir, save_top_k=1, save_last=True, mode=dl_kwargs['tracker_mode'])
    
#     early_stopping_callback = EarlyStopping(monitor=dl_kwargs['tracker_val'], min_delta=0.00, patience=5, mode=dl_kwargs['tracker_mode'])
    

#     model = SemanticSegmentationTask(**model_kwargs)
    
#     trainer = pl.Trainer(
#                 callbacks=[checkpoint_callback, early_stopping_callback],
#                 logger=logger,
#                 default_root_dir=experiment_dir,
#                 min_epochs=1,
#                 max_epochs=100,
#                 accelerator="gpu",
#                 devices=[2])
    
#     trainer.fit(model, trainloader, testloader)
    
    
    monitor_options = ['val_JaccardIndex']
    model_options = ['manet']
    encoder_options = ['resnet34']
    lr_options = [1e-4]
    loss_options = ["ce"]
    weight_init_options = ["imagenet"]
    in_channel = 4
    out_channel = 3

    for (model, encoder, lr, loss, weight_init, monitor_state) in itertools.product(
            model_options,
            encoder_options,
            lr_options,
            loss_options,
            weight_init_options,
            monitor_options):

        experiment_name = f"{monitor_state}_{model}_{encoder}_{lr}_{loss}_{weight_init}_None_loss"

        print(experiment_name)

        experiment_dir = os.path.join(dl_kwargs["out_dir"], experiment_name)
        logger = TensorBoardLogger(experiment_dir, name="models")

        if monitor_state == 'val_loss':
            tracking_mode = 'min'
        elif monitor_state == 'val_JaccardIndex':
            tracking_mode = 'max'

        checkpoint_callback = ModelCheckpoint(
            monitor=monitor_state, dirpath=experiment_dir, save_top_k=1, save_last=True, mode=tracking_mode)

        early_stopping_callback = EarlyStopping(monitor=monitor_state, min_delta=0.00, patience=5, mode=tracking_mode)


        model = SemanticSegmentationTask(
                        segmentation_model=model,
                        encoder_name=encoder,
                        encoder_weights=weight_init,
                        learning_rate=lr,
                        in_channels=in_channel,
                        num_classes=out_channel,
                        learning_rate_schedule_patience=6,
                        ignore_index=None,
                        loss=loss,
                        imagenet_pretraining=True)

        trainer = pl.Trainer(
                    callbacks=[checkpoint_callback, early_stopping_callback],
                    logger=logger,
                    default_root_dir=experiment_dir,
                    min_epochs=1,
                    max_epochs=100,
                    accelerator="gpu",
                    devices=[2])

        trainer.fit(model, trainloader, testloader)



