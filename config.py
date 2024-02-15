from dataclasses import dataclass, field
from dacite import from_dict #https://github.com/konradhalas/dacite
from typing import Optional
from pytorch_lightning.loggers import WandbLogger
import json 
from pathlib import Path
import yaml

@dataclass
class DatasetConfig:
    type: str #TODO convert to enum
    dir: str
    batch_size: int #TODO move this to datasetsconfig
    split_file: Optional[str] #TODO move this to datasetsconfig

    def get_keys(self, stage: str) -> list[str]:
        if self.split_file is None:
            print("No split file provided. Using all samples.")
            return None
        with open(self.split_file, 'r') as f:
            datastore = json.load(f)
            keys_to_use = datastore[stage]
        return keys_to_use

@dataclass
class DatasetsConfig:
    train: DatasetConfig
    validation: Optional[DatasetConfig]
    test: Optional[list[DatasetConfig]] # there can be multiple test datasets.

@dataclass
class TrainingParametersConfig:
	batch_size: int
	epochs: int
	init_lr: float
	lr_patience: int
	lr_decay_factor: float
 
@dataclass
class WandbConfig:
    #user_name: str
    project_name: str
    run_name: str

    def get_wanb_logger(self) -> WandbLogger:
        return  WandbLogger(project=self.project_name,
                                    name=self.run_name,
                                    log_model='all')



@dataclass
class WandbPreviousRunConfig:
	project_name: str
	run_id: str
	
	def get_wanb_logger(self) -> WandbLogger:
		return  WandbLogger(project=self.project_name,
									id=self.run_id,
									log_model='all')

@dataclass
class ModelConfig:
    loading_type: str #TODO change to enum 
    model_class: str
    loading_config: dict

@dataclass
class TrainConfig:
    datasets: DatasetsConfig
    wandb_config: Optional[WandbConfig]
    training_parameters: TrainingParametersConfig
    random_seed: int
    gpus_list: list[int]

@dataclass
class CVTrainConfig:
    wandb_config: Optional[WandbConfig]
    dataset_dir: int
    training_parameters: TrainingParametersConfig
    random_seed: int
    gpus_list: list[int]
    cv_folds: int

@dataclass
class TestConfig:
    test_datasets: list[DatasetConfig]
    model: ModelConfig
    wandb_config: Optional[WandbPreviousRunConfig]
    random_seed: int
    gpus_list: list[int]
    
def get_train_config(yaml_file_path: Path):
    config_yaml_path = Path(yaml_file_path)
    assert config_yaml_path.is_file()
    with open(config_yaml_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    config = from_dict(data_class=TrainConfig, data=config_dict)
    return config

def get_cv_train_config(yaml_file_path: Path):
    config_yaml_path = Path(yaml_file_path)
    assert config_yaml_path.is_file()
    with open(config_yaml_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    config = from_dict(data_class=CVTrainConfig, data=config_dict)
    return config

def get_test_config(yaml_file_path: Path):
    config_yaml_path = Path(yaml_file_path)
    assert config_yaml_path.is_file()
    with open(config_yaml_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    config = from_dict(data_class=TestConfig, data=config_dict)
    return config