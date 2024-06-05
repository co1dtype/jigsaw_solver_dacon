import os
import torch
import wandb
import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from datetime import datetime, timezone, timedelta

from modules.utils import load_yaml
from modules.trainer import LightningModel

import warnings
warnings.filterwarnings('ignore')

# Root Directory
PROJECT_DIR = os.path.dirname(__file__)

# Load config
config_path = os.path.join(PROJECT_DIR, 'config', 'train_config.yaml')
config = load_yaml(config_path)

# Seed
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True ## Segformer에서 Conv를 써서 True
seed_everything(config['TRAINER']['seed'], workers=True)

# GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(config['TRAINER']['gpu'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="Hyperparameter tuning for the neural network model.")

parser.add_argument("--batch_size", type=int, default=10, help="The number of samples per batch of computation. Default is 10.")
parser.add_argument("--epochs", type=int, default=80, help="The number of times the entire dataset is passed through the network. Default is 80.")
parser.add_argument("--learning_rate", type=float, default=3.235e-4, help="The step size for updating the network's weights. Default is 3.235e-4.")
parser.add_argument("--weight_decay", type=float, default=2.601e-4, help="The regularization parameter to prevent overfitting. Default is 2.601e-4.")

args = parser.parse_args()


if __name__ == "__main__":
    if config["LOGGER"]["wandb"]:
        CFG = {
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            "weight_decay": args.weight_decay,
            'batch_size': args.batch_size,
        }
        run = wandb.init(config=CFG)
        wandb.config.update(CFG)

        config["TRAINER"]["n_epoch"] = args.epochs
        config["TRAINER"]["learning_rate"] = args.learning_rate
        config["TRAINER"]["weight_decay"] = args.weight_decay
        config["DATALOADER"]["batch_size"] = args.batch_size

    if config['KFOLD']['kfold']:
        data_options = {'type':'k-fold', 
                        'path': config['KFOLD']['path'], 
                        'fold_number':config['KFOLD']['number']
                        }
    elif config['DATASET']['val_size'] > 0.0:
        data_options = {'type':'validation', 
                        'train_ratio':config['DATASET']['val_size']
                        }


    


    model = LightningModel(
        model_name = config["TRAINER"]["model"], 
        criterion= config["TRAINER"]["criterion"], 
        optimizer= config["TRAINER"]["optimizer"], 
        learning_rate= config["TRAINER"]["learning_rate"], 
        weight_decay= config["TRAINER"]["weight_decay"], 
        batch_size= config["DATALOADER"]["batch_size"], 
        num_workers= config["DATALOADER"]["num_workers"], 
        pin_memory= config["DATALOADER"]["pin_memory"], 
        seed= config["TRAINER"]["seed"], 
        shuffle= config["DATALOADER"]["shuffle"], 
        data_options = data_options, 
        centercutout= config["augumentations"]["CenterCutout"], 
        edgecutout = config["augumentations"]["EdgeCutout"]
        )
    
    early_stop_callback = EarlyStopping(    
        monitor= config["TRAINER"]["early_stopping_target"], 
        min_delta=0.001,     
        patience=config["TRAINER"]["early_stopping_patience"],          
        verbose=True,        
        mode= config["TRAINER"]["early_stopping_mode"]          
    )


    checkpoint_callback = ModelCheckpoint(
        monitor=config["TRAINER"]["early_stopping_target"],  
        dirpath= config["DIR"]['model_path'],
        filename= config["DIR"]['file_name'],
        save_top_k=1,       
        mode='max'         
    )

    trainer = Trainer(
        accelerator="gpu",
        max_epochs=config['TRAINER']['n_epochs'],
        deterministic=False,
        benchmark=True,
        callbacks=[early_stop_callback, checkpoint_callback]
    )

    model = model.to(device)
    trainer.fit(model)

