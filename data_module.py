from pytorch_lightning import LightningDataModule
from sklearn.model_selection import KFold
from typing import Optional

from torch.utils.data import Dataset, DataLoader
from data_loader import AugmentedXgazeDataset, XgazeDataset, OnlyAugmentedXgazeDataset

from pathlib import Path

class XgazeKFoldDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str = "data/", #train and validation data_dir should be the same
            split_seed: int = 12345,
            num_splits: int = 10,
            batch_size: int = 32,
            num_workers: int = 0,
            pin_memory: bool = False
        ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_splits = num_splits
        self.split_seed = split_seed
        
        #get all the files that end with .h5
        self.h5_files = [str(f) for f in Path(self.data_dir).glob('*.h5')]
        
        kf = KFold(n_splits=self.num_splits, shuffle=True, random_state=self.split_seed)
        self.all_splits = [k for k in kf.split(self.h5_files)]
        
        
        
    def setup(self, fold_number: int, train_dataset_type: str): #validation is always xgaze (without augmentation)
        train_indexes, val_indexes = self.all_splits[fold_number]
        train_keys, val_keys = [self.h5_files[i] for i in train_indexes], [self.h5_files[i] for i in val_indexes]
        
        if train_dataset_type == 'xgaze-with-augmented':
            self.data_train, self.data_val = AugmentedXgazeDataset(self.data_dir, train_keys, is_shuffle=True, is_load_label=True), XgazeDataset(self.data_dir, val_keys, is_shuffle=True, is_load_label=True)
        elif train_dataset_type == 'xgaze-only-augmented':
            self.data_train, self.data_val = OnlyAugmentedXgazeDataset(self.data_dir, train_keys, is_shuffle=True, is_load_label=True), XgazeDataset(self.data_dir, val_keys, is_shuffle=True, is_load_label=True)
        elif train_dataset_type == 'xgaze':
            self.data_train, self.data_val = XgazeDataset(self.data_dir, train_keys, is_shuffle=True, is_load_label=True), XgazeDataset(self.data_dir, val_keys, is_shuffle=True, is_load_label=True)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=True, shuffle=True)
        
        