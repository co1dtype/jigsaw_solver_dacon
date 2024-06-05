import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation
import time
import pandas as pd
import albumentations as A
import numpy as np
from glob import glob
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from modules.augmentations import CenterCutout, EdgeCutout

from modules.datasets import JigsawDataset
from modules.optimizers import get_optimizer
from modules.losses import get_loss
from models.utils import get_model
from modules.metrics import PRA
from pytorch_lightning import LightningModule



class LightningModel(LightningModule):
    def __init__(self, model_name, criterion, optimizer, learning_rate, weight_decay, batch_size, 
                 num_workers, pin_memory, seed, shuffle, data_options, centercutout, edgecutout):
        super().__init__()
        self.save_hyperparameters()

        self.model = get_model(model_name)
        self.criterion = get_loss(criterion)
        self.optimizer = get_optimizer(optimizer)
        self.optimizer = self.optimizer(params=self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.shuffle = shuffle
        self.data_options = data_options
        self.centercutout = centercutout
        self.edgecutout = edgecutout
        self.PRA = PRA()

        self.train_time = 0
        self.val_time = 0
        self.train_logs = None
        self.val_logs = None

        segformer = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
        segformer.decode_head.classifier = nn.Conv2d(768, 16, kernel_size=(1, 1), stride=(1, 1))
        
        self.segformer = segformer
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.reduce = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, inputs):
        return self.model(inputs)
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        output = self(inputs)

        torch.use_deterministic_algorithms(False)
        loss = self.criterion(output, labels)
        torch.use_deterministic_algorithms(True)
        
        self.PRA(output, labels)

        self.log('train_PRA', self.PRA.compute())  
        self.log('train_loss', loss)  
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        output = self(inputs)
        torch.use_deterministic_algorithms(False)
        loss = self.criterion(output, labels)
        torch.use_deterministic_algorithms(True)
        self.PRA(output, labels)

        self.log('val_PRA', self.PRA.compute())  
        self.log('val_loss', loss) 
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch
        return self(x)    
    
    def on_train_epoch_start(self):
        self.train_time = time.time()

    def on_train_epoch_end(self):
        self.train_time = time.time() - self.train_time
        self.train_logs = self.PRA.get_accuracies()
        self.PRA.reset()

    def on_validation_epoch_start(self):
        self.val_time = time.time()

    def on_validation_epoch_end(self):
        self.val_time = time.time() - self.val_time
        self.val_logs = self.PRA.get_accuracies()
        self.PRA.reset()

    def configure_optimizers(self):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8)
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',  
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    ####################
    # DATA RELATED HOOKS
    ####################
    def prepare_data(self) -> None:
        train_df = pd.read_csv('./train.csv')
        self.test_df = pd.read_csv('./test.csv')

        if self.data_options['type'] == "k-fold":
            kfold = np.load(self.data_options['path'])
            train_ids = kfold['train_ids_list'][self.data_options['fold_number']]
            val_ids = kfold['val_ids_list'][self.data_options['fold_number']]

            self.val_df = train_df.iloc[val_ids]
            self.train_df = train_df.iloc[train_ids]

            self.train_labels = self.train_df.iloc[:, 2:].values.reshape(-1, 4, 4)
            self.val_labels = self.val_df.iloc[:, 2:].values.reshape(-1, 4, 4)

        elif self.data_options['type'] == "validation":
            train_len = int(len(train_df) * self.data_options["train_ratio"])
            self.val_df = train_df.iloc[train_len:]
            self.train_df = train_df.iloc[:train_len]
        
            self.train_labels = self.train_df.iloc[:, 2:].values.reshape(-1, 4, 4)
            self.val_labels = self.val_df.iloc[:, 2:].values.reshape(-1, 4, 4)

        else:
            self.train_df = train_df
            self.train_labels = self.train_df.iloc[:, 2:].values.reshape(-1, 4, 4)
            self.val_df = None
            self.val_labels = None


    def get_train_transform(self):
        transform_list = [
            A.RandomBrightnessContrast(p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
                A.ChannelShuffle(p=1),
                A.CLAHE(p=1.0)
            ], p=1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]

        if self.centercutout:
            transform_list.insert(2, CenterCutout(p=1))
        if self.edgecutout:
            transform_list.insert(2, EdgeCutout(p=1))
        return A.Compose(transform_list)
    
    def get_test_transform(self):
        test_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        return test_transform

    def setup(self, stage=None):
        train_transform = self.get_train_transform()
        test_transform = self.get_test_transform()

        self.jigsaw_train = JigsawDataset(self.train_df, self.train_labels, train_transform)
        if self.val_df is not None:
            self.jigsaw_val = JigsawDataset(self.val_df, self.val_labels, test_transform)
        self.jigsaw_test = JigsawDataset(self.test_df, None, test_transform)
    
    def train_dataloader(self):
        return DataLoader(self.jigsaw_train, batch_size=self.batch_size, shuffle=self.shuffle, 
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        if self.val_df is not None:
            return DataLoader(self.jigsaw_val, batch_size=self.batch_size, shuffle=False, 
                              num_workers=self.num_workers, pin_memory=self.pin_memory)
        else:
            return None

    def test_dataloader(self):
        return DataLoader(self.jigsaw_test, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def predict_dataloader(self):
        return DataLoader(self.jigsaw_test, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, pin_memory=self.pin_memory)
