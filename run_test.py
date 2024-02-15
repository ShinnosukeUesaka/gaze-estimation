import torch
from trainer import Trainer
#from config import get_config
from data_loader import get_data_loader
from config import get_test_config

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

model_dict = {
	'GazeModel': GazeModel
}

if __name__ == '__main__':

	#config, unparsed = get_config()

	config_yaml_path = Path(sys.argv[1])

	config = get_test_config(config_yaml_path)

	wandb_logger = None
	if config.wandb_config is not None:
		wandb_logger = config.wandb_config.get_wanb_logger()

	set_seed(config.random_seed)



	if config.model.loading_type == 'wandb':
		assert wandb_logger is not None
		reference = f"{wandb_logger.experiment.entity}/{config.model.loading_config['project_name']}/model-{config.model.loading_config['run_id']}:best_k"
		checkpoint_path = wandb_logger.download_artifact(artifact=reference, artifact_type='model')
	elif config.model.loading_type == 'local_path':
		checkpoint_path = config.model.loading_config['path']
	else: 
		raise NotImplementedError

	checkpoint_path = Path(checkpoint_path) / 'model.ckpt'

	model = model_dict[config.model.model_class].load_from_checkpoint(checkpoint_path=checkpoint_path)
	trainer = pl.Trainer(logger=wandb_logger, accelerator="gpu", devices=(config.gpus_list))

	test_data_loaders = []

	for i, test_dataset in enumerate(config.test_datasets):

		test_data_loader = get_data_loader(
				test_dataset.type, # for different loading pattern
				test_dataset.dir, 
				test_dataset.batch_size,
				is_load_label=True,
				num_workers = NUM_WORKERS,
				is_shuffle=False,
				keys_to_use=test_dataset.get_keys('test')
		)
		images, captions = test_data_loader.dataset.get_image_samples_and_captions(NUMBER_OF_SAMPLES)
		wandb_logger.log_image(key=f"Test Samples{i}:{test_dataset.type}", images=images, caption=captions)
		test_data_loaders.append(test_data_loader)

	
	trainer.test(model, dataloaders=test_data_loaders) #,the best model checkpoint from the previous trainer.fit call will be loaded.