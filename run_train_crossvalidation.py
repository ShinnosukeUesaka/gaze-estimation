import torch
from trainer import Trainer
#from config import get_config
from data_loader import get_data_loader
from config import get_cv_train_config
from data_module import XgazeKFoldDataModule

from model import GazeModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from dataclasses import dataclass, field
from dacite import from_dict #https://github.com/konradhalas/dacite


import wandb
import numpy as np
import json
import os
from typing import Optional


import os.path as osp
import random
import yaml
import sys
from pathlib import Path

NUMBER_OF_SAMPLES = 40
NUM_WORKERS = 5

def set_seed(seed):
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	# ensure reproducibility
	os.environ["PYTHONHASHSEED"] = str(seed)



    


if __name__ == '__main__':

	#config, unparsed = get_config()

	config_yaml_path = Path(sys.argv[1])
	config = get_cv_train_config(config_yaml_path)
	
	do_testing = config.datasets.test is not None

	if do_testing:
		raise('testing is not supported yet')

	if config.wandb_config is not None:
		wandb_logger = WandbLogger(project=config.wandb_config.project_name,
									name=config.wandb_config.run_name,
									log_model='all')
		
		wandb_logger.experiment.config.update(
			config
		)
		wandb.define_metric(name='train_loss', summary='best', goal='minimize')
		wandb.define_metric(name='val_loss', summary='best', goal='minimize')

		wandb.define_metric(name='train_angular_error', summary='best', goal='minimize')
		wandb.define_metric(name='val_angular_error', summary='best', goal='minimize')
	



	set_seed(config.random_seed)

	# keys_to_use_train = config.datasets.train.get_keys('train') # Use all the files in the dataset directory
	# keys_to_use_validation = config.datasets.validation.get_keys('validation')
	
	dm = XgazeKFoldDataModule(data_dir=config.dataset_dir, split_seed=config.random_seed, num_splits=config.cv_folds, batch_size=32,
                           num_workers=NUM_WORKERS, pin_memory=True)
	
	for k in range(config.cv_folds):
		dm.setup(fold_number=k, train_dataset_type='xgaze')


	
	
	images, captions = train_data_loader.dataset.get_image_samples_and_captions(NUMBER_OF_SAMPLES)
	wandb_logger.log_image(key="Train Samples", images=images, caption=captions)

	images, captions = validation_data_loader.dataset.get_image_samples_and_captions(NUMBER_OF_SAMPLES)
	wandb_logger.log_image(key="Validation Samples", images=images, caption=captions)


	model = GazeModel(lr=config.training_parameters.init_lr, lr_patience=config.training_parameters.lr_patience, lr_decay_factor=config.training_parameters.lr_decay_factor)

	wandb_logger.watch(model=model)
 
	checkpoint_callback = ModelCheckpoint(monitor="val_angular_error", mode="max")
	trainer = pl.Trainer(logger=wandb_logger, accelerator="gpu", devices=(config.gpus_list), max_epochs=config.training_parameters.epochs, callbacks=[checkpoint_callback])

	trainer.fit(model, train_data_loader, validation_data_loader)
	
	if not do_testing:
		sys.exit()
  
	# test_data_loaders = []

	# for i, test_dataset in enumerate(config.datasets.test):

	# 	test_data_loader = get_data_loader(
	# 			test_dataset.type, # for different loading pattern
	# 			test_dataset.dir, 
	# 			test_dataset.batch_size,
	# 			is_load_label=True,
	# 			num_workers = NUM_WORKERS,
	# 			is_shuffle=True,
	# 			keys_to_use=test_dataset.get_keys('test')
	# 	)
	# 	images, captions = test_data_loader.dataset.get_image_samples_and_captions(NUMBER_OF_SAMPLES)
	# 	wandb_logger.log_image(key=f"Test Samples{i}:{test_dataset.type}", images=images, caption=captions)
	# 	test_data_loaders.append(test_data_loader)

	
	# trainer.test(dataloaders=test_data_loaders) #,the best model checkpoint from the previous trainer.fit call will be loaded.