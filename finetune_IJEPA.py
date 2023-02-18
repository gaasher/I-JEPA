import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
)
from pytorch_lightning.loggers import WandbLogger
from model import IJEPA_base
from pretrain_IJPEA import IJEPA


'''Dummy Dataset'''
class IJEPADataset(Dataset):
    def __init__(self,
                 dataset_path,
                 stage='train',
                 ):
        super().__init__()
        img1 =torch.randn(3, 224, 224)
        self.data = img1.repeat(100, 1, 1, 1)
        label = torch.tensor([0., 0., 0., 1., 0.])
        self.label = label.repeat(100, 1)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
'''
Placeholder for datamodule in pytorch lightning
'''
class D2VDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_path,
                 batch_size=16,
                 num_workers=4,
                 pin_memory=True,
                 shuffle=True
                 ):
        super().__init__()
        
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        
    def setup(self, stage=None):
        self.train_dataset = IJEPADataset(dataset_path=self.dataset_path, stage='train')
        self.val_dataset = IJEPADataset(dataset_path=self.dataset_path, stage='val')
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

'''
Finetune IJEPA
'''
class IJEPA_FT(pl.LightningModule):
    #take pretrained model path, number of classes, learning rate, weight decay, and drop path as input
    def __init__(self, pretrained_model_path, num_classes, lr=1e-3, weight_decay=0, drop_path=0.1):

        super().__init__()
        self.save_hyperparameters()

        #set parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.drop_path = drop_path

        #define model layers
        self.pretrained_model = IJEPA.load_from_checkpoint(pretrained_model_path)
        self.pretrained_model.model.mode = "test"
        self.pretrained_model.model.layer_dropout = self.drop_path
        self.average_pool = nn.AvgPool1d((self.pretrained_model.embed_dim), stride=1)
        #mlp head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.pretrained_model.num_tokens),
            nn.Linear(self.pretrained_model.num_tokens, num_classes),
        )

        #define loss
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.pretrained_model.model(x)
        x = self.average_pool(x) #conduct average pool like in paper
        x = x.squeeze(-1)
        x = self.mlp_head(x) #pass through mlp head
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y) #calculate loss
        accuracy = (y_hat.argmax(dim=1) == y.argmax(dim=1)).float().mean() #calculate accuracy
        self.log('train_accuracy', accuracy)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        accuracy = (y_hat.argmax(dim=1) == y.argmax(dim=1)).float().mean()
        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        return self(batch[1])
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

if __name__ == '__main__':
    dataset = D2VDataModule(dataset_path='data')

    model = IJEPA_FT(pretrained_model_path='.ckpt', num_classes=5)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    trainer = pl.Trainer(
        accelerator='cpu',
        precision=16,
        max_epochs=10,
        callbacks=[lr_monitor, model_summary],
        gradient_clip_val=.1,
    )

    trainer.fit(model, dataset)