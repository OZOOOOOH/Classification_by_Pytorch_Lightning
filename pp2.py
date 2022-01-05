import torchmetrics
import torchvision.utils

import wandb
from cv2 import cv2
import timm
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from albumentations.core.composition import Compose, OneOf
from albumentations.augmentations.transforms import CLAHE, GaussNoise, ISONoise
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from torchmetrics.functional import accuracy
from torchmetrics.functional import confusion_matrix as conf_mat
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import LightningDataModule
from pytorch_lightning import LightningModule
import os
from multiprocessing import freeze_support

## Initialize wandb logger
wandb.init(project="mnist")

CONFIG = dict(
    seed=42,
    train_val_split=0.2,
    model_name="efficientnet_b0",
    pretrained=True,
    img_size=256,
    num_classes=3,
    lr=5e-4,
    min_lr=1e-6,
    t_max=20,
    num_epochs=20,
    batch_size=32,
    accum=1,
    precision=16,
    n_fold=5,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
# Seed everything
seed_everything(CONFIG["seed"])


def bring_dataset_csv(datatype='CT',stage=None):

    # Directories
    PATH = f"/media/quiil/data1/data/COVID_{datatype}/"

    # Read CSV file
    if stage == "fit" or stage is None:
        df_train = pd.read_csv(PATH + "metadata_train.csv")
        df_val = pd.read_csv(PATH + "metadata_val.csv")
        return df_train, df_val

    else:
        df_test = pd.read_csv(PATH + "metadata_test.csv")
        return df_test

def making_wandb_table_cols():
        # columns = ["id", "image", "guess", "truth"]
        columns = ["image", "guess", "truth"]
        #TODO consider brining image_path
        classes=['Normal', 'COVID', 'Others']
        #TODO argparser here
        for cls in classes:
            columns.append("score_" + cls)
        return columns


def log_test_predictions(images, targets, outputs, preds, test_table):
    # obtain confidence scores for all classes
    scores = F.softmax(outputs.data, dim=-1)
    images=images.view([-1, 3, 256, 256])
    scores=scores.view([-1,3])
    log_scores = scores.cpu().numpy()
    log_images = images.cpu().numpy()
    log_labels = targets.cpu().numpy()
    log_preds = preds.cpu().numpy()

    all_batches=log_images.shape[0]

    print(f'log_images:{log_images.shape}')
    print(f'log_images:{log_images}')
    # adding ids based on the order of the images
    _id = 0


    for i, l, p, s in zip(log_images, log_labels, log_preds, log_scores):
        # add required info to data table:
        # id, image pixels, model's guess, true label, scores for all classes
        # img_id = str(_id) + "_" + str(log_counter)
        print(f'\ni.shape:{i.shape}\n')
        print(f'i:{i}')

        i = np.transpose(i, (1, 2, 0))
        # test_table.add_data(img_id, wandb.Image(i), p, l, *s)
        test_table.add_data(wandb.Image(i), p, l, s[0],s[1],s[2])

        _id += 1
        if _id == CONFIG['batch_size']:
            break


class CustomNet(nn.Module):
    def __init__(self, model_name="resnet50", pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=CONFIG['num_classes'])
    def forward(self, x):
        x = self.model(x)
        return x

class Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.image_id = df["img"].values
        self.labels = df["label"].values
        self.transform = transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        image_path = self.image_id[idx]
        label = self.labels[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return {"img": image, "target": label, "id": image_path}

class DataModule(LightningDataModule):
    def __init__(self, batch_size, data_dir: str = "./"):
        super().__init__()
        self.batch_size = batch_size

        # Train augmentation policy
        self.train_transform = Compose(
            [
                A.RandomResizedCrop(height=CONFIG['img_size'], width=CONFIG['img_size']),
                A.HorizontalFlip(p=0.5),
                # Flip the input horizontally around the y-axis.
                A.ShiftScaleRotate(p=0.5),
                # Randomly apply affine transforms: translate, scale and rotate the input
                A.RandomBrightnessContrast(p=0.5),
                # Randomly change brightness and contrast of the input image.
                A.Normalize(),
                # Normalization is applied by the formula: img = (img - mean * max_pixel_value) / (std * max_pixel_value)
                ToTensorV2(),
                # Convert image and mask to torch.Tensor
            ]
        )
        # Validation/Test augmentation policy
        self.test_transform = Compose(
            [
                A.Resize(height=CONFIG['img_size'], width=CONFIG['img_size']),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
    def setup(self, stage=None):
        # Assign train/val/test datasets for use in dataloaders
        if stage == "fit" or stage is None:
            # Random train-validation split
            train_df ,valid_df=bring_dataset_csv(datatype='CT',stage=None)
            #TODO argparser here

            # Train dataset
            self.train_dataset = Dataset(train_df, self.train_transform)
            # Validation dataset
            self.valid_dataset = Dataset(valid_df, self.test_transform)
            # Test dataset
        else:
            test_df = bring_dataset_csv(datatype='CT', stage='test')
            self.test_dataset = Dataset(test_df, self.test_transform)
        print("setup done!")
    def train_dataloader(self):
        print("Train Data loading")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
        )
    def val_dataloader(self):
        print("Val data loading")
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )
    def test_dataloader(self):
        print("Test data loading")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            drop_last=True,
        )

class LitModel(LightningModule):
    def __init__(self, model):
        super(LitModel, self).__init__()
        self.model = model
        self.metric = torchmetrics.Accuracy()
        self.criterion = nn.CrossEntropyLoss()
        self.lr = CONFIG["lr"]

    def forward(self, x):
        output=self.model(x)
        return output

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=CONFIG["t_max"],
            eta_min=CONFIG["min_lr"]
        )

        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def training_step(self, batch, batch_idx):
        image,target = batch["img"], batch["target"]
        output = self(image)
        preds=output.argmax(1)
        # print(f'In training_step, output shape: {output.shape}')
        loss = self.criterion(output, target)
        score = self.metric(preds=preds, target=target)

        # metrics -> accuracy

        logs = {
            "train_loss": loss,
            "train_accuracy": score,
        }
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        print('\nWe are in validation_step!!!\n')
        image,target = batch["img"], batch["target"]
        output = self(image)
        # print(f"In validation_step, output shape: {output.shape}")
        # print(f"target: {target}")
        preds = output.argmax(1)
        loss = self.criterion(output, target)
        score = self.metric(preds=preds, target=target)
        # confusion_matrix = torch.zeros(3, 3)
        #
        # for t, p in zip(target.view(-1), output.argmax(1).view(-1)):
        #         confusion_matrix[t.long(), p.long()] += 1
        # print("confusion_matrix:\n",confusion_matrix)

        logs = {
            "valid_loss": loss,
            "valid_accracy": score,
            # "confusion_matrix": confusion_matrix
                }

        self.log_dict(logs,
                      on_step=False, on_epoch=True, prog_bar=True, logger=True
                      )
        return {"valid_loss": loss, "valid_accracy": score, "preds": preds, "target": target,"image":image,"output":output}

    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]
        avg_loss = torch.stack([x['valid_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['valid_accracy'] for x in outputs]).mean()

        preds = torch.cat([x['preds'] for x in outputs])
        targets = torch.cat([x['target'] for x in outputs])
        images = torch.stack([x['image'] for x in outputs])
        outputs_s = torch.stack([x['output'] for x in outputs])

        print(f'\nIn validation_epoch_end,targets:{targets}')

        confusion_matrix = conf_mat(preds, targets, num_classes=3)
        #TODO argparser here
        print(f"\nIn validation_epoch_end, confusion_matrix\n {confusion_matrix}")

        wandb.log({
            "Val Confusion Matrix": wandb.plot.confusion_matrix(preds=preds.cpu().numpy(),
                                                                y_true=targets.cpu().numpy(),
                                                                class_names=['Normal', 'COVID', 'Others'])
            #TODO argparser here class_names
            #TODO figure out how to add steps
        })

        test_table = wandb.Table(columns=making_wandb_table_cols())

        #TODO set the log_counter

        # if log_counter < 10:
        log_test_predictions(images, targets, outputs_s, preds, test_table)
            # log_counter += 1

        wandb.log({"Val Table": test_table})

        return {'avg_valid_loss': avg_loss,
                'avg_valid_accracy': avg_acc,
                'preds':preds,
                'targets':targets
                }

    def test_step(self, batch, batch_idx) :
        print('\nWe are in test_step!!!\n')
        image,target = batch["img"], batch["target"]
        output = self(image)
        # print(f"In validation_step, output shape: {output.shape}")
        # print(f"target: {target}")
        preds = output.argmax(1)
        loss = self.criterion(output, target)
        score = self.metric(preds=preds, target=target)

        logs = {
            "valid_loss": loss,
            "valid_accracy": score,
            # "confusion_matrix": confusion_matrix
                }

        self.log_dict(logs,
                      on_step=False, on_epoch=True, prog_bar=True, logger=True
                      )
        return {"valid_loss": loss, "valid_accracy": score, "preds": preds, "target": target,"image":image,"output":output}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['valid_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['valid_accracy'] for x in outputs]).mean()

        preds = torch.cat([x['preds'] for x in outputs])
        targets = torch.cat([x['target'] for x in outputs])
        images = torch.stack([x['image'] for x in outputs])
        outputs_s = torch.stack([x['output'] for x in outputs])
        # print(f'\ntargets:{targets}')

        confusion_matrix = conf_mat(preds, targets, num_classes=3)
        #TODO argparser here
        print(f"\nTest confusion_matrix\n {confusion_matrix}")
        # df_cm=pd.DataFrame(confusion_matrix.numpy(),index=range(3),columns=range(3))
        # logs = {'avg_test_loss': avg_loss,
        #         'avg_valid_accracy': avg_acc,
        #         # "valid_confusion_matrix": confusion_matrix
        #         }
        # self.log_dict(logs,
        #               on_step=False, on_epoch=True, prog_bar=True, logger=True
        #               )
        # wandb.sklearn.plot.confusion_matrix(preds.cpu().numpy(),targets.cpu().numpy(), labels=range(3))
        wandb.log({
            "Test Confusion Matrix": wandb.plot.confusion_matrix(preds=preds.cpu().numpy(),
                                                                y_true=targets.cpu().numpy(),
                                                                class_names=['Normal', 'COVID', 'Others'])
            #TODO fix class_names
        })

        test_table = wandb.Table(columns=making_wandb_table_cols())

        #TODO set the log_counter

        # if log_counter < 10:
        log_test_predictions(images, targets, outputs_s, preds, test_table)
            # log_counter += 1

        wandb.log({"Test Table": test_table})

        return {'avg_test_loss': avg_loss,
                'avg_test_accracy': avg_acc,
                'preds':preds,
                'targets':targets
                }

# Checkpoint
checkpoint_callback = ModelCheckpoint(
    dirpath='xray/path/',
    #TODO argparser here
    monitor="valid_accracy",
    # 어떤 metric을 기준으로 체크포인트를 저장할지
    mode="max",
    save_top_k=1,
    # 최대 몇 개의 체크포인트를 저장할지
    save_last=True,
    # 마지막 체크포인트를 저장
    save_weights_only=True,
    filename="checkpoint/{epoch:02d}-{valid_loss:.4f}-{valid_accracy:.4f}",
    verbose=True,
    # 체크포인트 저장 결과를 출력
)
# Earlystopping
earlystopping = EarlyStopping(
    monitor="valid_accracy",
    patience=3,
    mode="max",
    verbose=True
)


# Custom Callback
class ImagePredictionLogger(Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples["img"], val_samples["target"]

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        print('\nWe are in on_validation_epoch_end!!!!')
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        print(f'len of val_labels: {len(val_labels)}')
        # Get model prediction
        logits = pl_module(val_imgs)

        print(f'logits.shape: {logits.shape}')

        preds = torch.argmax(logits, -1)
        # Log the images as wandb Image

        confusion_matrix = torch.zeros(3, 3)

        for t, p in zip(val_labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

        print("confusion_matrix:\n",confusion_matrix)



        columns = ["image", "guess", "truth"]
        for cls in ['Normal', 'COVID', 'Others']:
            columns.append("score_" + cls)



        test_table=wandb.Table(columns=columns).add_data(
                wandb.Image(val_imgs),
                preds,
                val_labels,
                logits[0],
                logits[1],
                logits[2],

            )

        wandb.log(
            {
                "examples": [
                    wandb.Image(x, caption=f"Pred:{pred}, True:{y}")
                    for x, pred, y in zip(
                        val_imgs,
                        preds,
                        val_labels,
                    )
                ],
                # "test_predictions":[
                #     wandb.Table(columns=columns).add_data(
                #         wandb.Image(val_imgs),
                #         preds,
                #         val_labels,
                #         logits[0],
                #         logits[1],
                #         logits[2],
                #
                #     )
                # ]
                "val_table": test_table,

            },
            commit=False,
            #TODO Figure out what does the commit do
        )
#

#TODO Separate these to py files

# When logging manually through wandb.log or trainer.logger.experiment.log, make sure to use commit=False so the logging step does not increase.

# Init our data pipeline
datamodule = DataModule(batch_size=CONFIG["batch_size"])
datamodule.setup()

# Samples required by the custom ImagePredictionLogger callback to log image predictions.
val_samples = next(iter(datamodule.val_dataloader()))
val_imgs, val_labels = val_samples["img"], val_samples["target"]

# Init our model

model = CustomNet(model_name=CONFIG["model_name"], pretrained=CONFIG["pretrained"])

lit_model = LitModel(model)

## Initialize wandb logger
wandb_logger = WandbLogger(
    project="dann", config=CONFIG,  job_type="train"
)


# lr_monitor_step=LearningRateMonitor(logging_interval='step')
lr_monitor_epoch=LearningRateMonitor(logging_interval='epoch')
# Initialize a trainer
trainer = Trainer(
    max_epochs=CONFIG["num_epochs"],
    # gpus=[0,1],
    gpus=[1],
    accumulate_grad_batches=CONFIG["accum"],
    precision=CONFIG["precision"],
    auto_lr_find=True,
    # checkpoint_callback=checkpoint_callback,
    # callbacks=[earlystopping,checkpoint_callback,lr_monitor_epoch,ImagePredictionLogger(val_samples,32)],
    callbacks=[earlystopping,checkpoint_callback,lr_monitor_epoch],
    # callbacks=[earlystopping,
    #            ImagePredictionLogger(val_samples)
    #            ],
    # checkpoint_callback=checkpoint_callback,
    # strategy='dp',
    logger=wandb_logger,
    weights_summary="top",
    num_sanity_val_steps=0,
)
wandb_logger.watch(lit_model)
#TODO: train 후 validation 중 progress bar 수정

print(f'#####################The Network model name is {CONFIG["model_name"]}####################')


trainer.tune(lit_model,datamodule.train_dataloader(),datamodule.val_dataloader())
lr_finder=trainer.tuner.lr_find(lit_model,datamodule.train_dataloader(),datamodule.val_dataloader())
new_lr=lr_finder.suggestion()
lit_model.hparams.lr=new_lr

# fig=lr_finder.plot(suggest=True)
# fig.show()
#TODO make func this
#
# Train the model ⚡🚅⚡
trainer.fit(lit_model, datamodule.train_dataloader(),datamodule.val_dataloader())
#
#
# loaded_model=lit_model.load_from_checkpoint()
#
# trainer.test(lit_model,dataloaders=datamodule.test_dataloader(),ckpt_path='/home/quiil/PycharmProjects/NewCOVID_Project/xray/path/checkpoint/epoch=05-valid_loss=0.1438-valid_accracy=0.9530.ckpt',)

# Close wandb run
wandb.finish()

