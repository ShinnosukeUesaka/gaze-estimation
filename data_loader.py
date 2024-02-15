import numpy as np
import h5py
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import json
import random
from typing import List
import cv2
from pathlib import Path

trans_train = transforms.Compose([
		transforms.ToPILImage(),
		transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
		transforms.Normalize(mean=[0.485, 0.456, 0.406],
							 std=[0.229, 0.224, 0.225]),
	])

trans = transforms.Compose([
		transforms.ToPILImage(),
		transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
		transforms.Normalize(mean=[0.485, 0.456, 0.406],
							 std=[0.229, 0.224, 0.225]),
	])

hi = torch.utils.data



def get_data_loader(data_name, data_dir, 
						   batch_size,
							is_load_label: True,
						   num_workers=4,
						   is_shuffle=True,
         					keys_to_use: bool=None):
	# there are three subsets for ETH-XGaze dataset: train, test and test_person_specific
	# train set: the training set includes 80 participants data
	# test set: the test set for cross-dataset and within-dataset evaluations
	# test_person_specific: evaluation subset for the person specific setting
	
	if data_name == 'xgaze':
		train_set = XgazeDataset(dataset_path=data_dir, keys_to_use=keys_to_use,
						transform=None, is_shuffle=is_shuffle, is_load_label=is_load_label)
	
	elif data_name == 'xgaze-with-augmented':
		train_set = AugmentedXgazeDataset(dataset_path=data_dir, keys_to_use=keys_to_use,
						transform=None, is_shuffle=is_shuffle, is_load_label=is_load_label)
	
	elif data_name == 'xgaze-only-augmented':
		train_set = OnlyAugmentedXgazeDataset(dataset_path=data_dir, keys_to_use=keys_to_use,
						transform=None, is_shuffle=is_shuffle, is_load_label=is_load_label)
	
	elif data_name == 'mpii_normalized':
			train_set = MpiiDataset(dataset_path=data_dir, keys_to_use=keys_to_use,
					transform=None, is_shuffle=is_shuffle, is_load_label=is_load_label)

	else:
		raise Exception('Data name does not exist')

	data_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True, shuffle=is_shuffle)


	print(f"Number of images: {len(train_set)}")
	return data_loader








class XgazeDataset(Dataset):
	def __init__(self, dataset_path: str, keys_to_use: List[str] = None, transform=None, is_shuffle=True,
				 is_load_label=True):
		self.dir_path = Path(dataset_path)
		self.is_load_label = is_load_label
		self.transform = transform
		self.keys_to_use = keys_to_use

		if self.keys_to_use == None:
			h5_files = self.dir_path.glob('*.h5')
			self.keys_to_use = [h5_file.name for h5_file in h5_files]
   

		assert len(self.keys_to_use) > 0

		self.image_locations = []

		for key in self.keys_to_use:
			file_path = self.dir_path / key
			assert file_path.is_file()
			h5_file = h5py.File(file_path, 'r', swmr=True)
			for n in range(h5_file['face_patch'].shape[0]):
				self.image_locations.append(
					{
					'h5_file_name': key,
					'dataset_name': 'face_patch',
					'label_dataset_name': 'face_gaze',
					'image_index': (n),
					'label_index': (n),
					}	
				)
			h5_file.close()

		if is_shuffle:
			print('shuffle')
			random.shuffle(self.image_locations)  # random the order to stable the training
  
	def __len__(self):
		return len(self.image_locations)

	def __del__(self):
		pass

	def __getitem__(self, idx):
		image_location =  self.image_locations[idx]
		hdf = h5py.File(self.dir_path / image_location['h5_file_name'], 'r', swmr=True)
		assert hdf.swmr_mode

		# Get face image
		image = hdf[image_location['dataset_name']][image_location['image_index']]
		image = image[:, :, [2, 1, 0]]  # from BGR to RGB
		image = trans(image)
		
		if self.transform:
			image = self.transform(image)
		
		# Get labels
		if self.is_load_label:
			gaze_label = hdf[image_location['label_dataset_name']][image_location['label_index']]
			gaze_label = gaze_label.astype('float')
			return image, gaze_label
		else:
			return image

	def get_image_samples_and_captions(self, number_of_samples):
		assert number_of_samples < len(self)
    
		sample_index = random.sample(range(len(self)), number_of_samples)
		
		images = []
		captions = []
		if self.is_load_label:
			for i in sample_index:
				images.append(self[i][0])
				captions.append(f"Gaze Direction: {self[i][1]}")
		else: 
			for i in sample_index:
				images.append(self[i][0])
				captions.append(f"Face Patch")
			
		return images, captions


class AugmentedXgazeDataset(XgazeDataset):
	"""Load augmented and original images"""
	def __init__(self, dataset_path: str, keys_to_use: List[str] = None, transform=None, is_shuffle=True, is_load_label=True):
		super().__init__(dataset_path, keys_to_use, transform, is_shuffle, is_load_label)

		for key in self.keys_to_use:
			file_path = self.dir_path / key
			assert file_path.is_file()

			h5_file = h5py.File(file_path, 'r', swmr=True)
			for n in range(h5_file['augmented_face_patch'].shape[0]):
				for m in range(h5_file['augmented_face_patch'].shape[1]):
					self.image_locations.append(
					{
					'h5_file_name': key,
					'dataset_name': 'augmented_face_patch',
					'label_dataset_name': 'face_gaze',
					'image_index': (n,m),
					'label_index': (n),
					}	
				)
			h5_file.close()
		
		if is_shuffle:
			print('shuffle')
			random.shuffle(self.image_locations)
   
	def image_is_augmented(self, index):
		return self.image_locations[index]['dataset_name'] == 'augmented_face_patch'

	def get_image_samples_and_captions(self, number_of_samples):
		assert number_of_samples < len(self)
    
		sample_index = random.sample(range(len(self)), number_of_samples)
		
		images = []
		captions = []
		if self.is_load_label:
			for i in sample_index:
				images.append(self[i][0])
				captions.append(f"Augmented: {self.image_is_augmented(i)}. Gaze Direction: {self[i][1]}")
		else: 
			for i in sample_index:
				images.append(self[i][0])
				captions.append(f"Augmented: {self.image_is_augmented(i)}")
			
		return images, captions


class OnlyAugmentedXgazeDataset(XgazeDataset):
	"""Only load augmented images"""
	def __init__(self, dataset_path: str, keys_to_use: List[str] = None, transform=None, is_shuffle=True, is_load_label=True):
		super().__init__(dataset_path, keys_to_use, transform, is_shuffle, is_load_label)

		self.image_locations = [] # delete image locations (not augmented) provided from the parent class.

		for key in self.keys_to_use:
			file_path = self.dir_path / key
			assert file_path.is_file()

			h5_file = h5py.File(file_path, 'r', swmr=True)
			for n in range(h5_file['augmented_face_patch'].shape[0]):
				for m in range(h5_file['augmented_face_patch'].shape[1]):
					self.image_locations.append(
					{
					'h5_file_name': key,
					'dataset_name': 'augmented_face_patch',
					'label_dataset_name': 'face_gaze',
					'image_index': (n,m),
					'label_index': (n),
					}	
				)
			h5_file.close()
		
		if is_shuffle:
			print('shuffle')
			random.shuffle(self.image_locations)
   
	def image_is_augmented(self, index):
		return self.image_locations[index]['dataset_name'] == 'augmented_face_patch'

	def get_image_samples_and_captions(self, number_of_samples):
		assert number_of_samples < len(self)
    
		sample_index = random.sample(range(len(self)), number_of_samples)
		
		images = []
		captions = []
		if self.is_load_label:
			for i in sample_index:
				images.append(self[i][0])
				captions.append(f"Augmented: {self.image_is_augmented(i)}. Gaze Direction: {self[i][1]}")
		else: 
			for i in sample_index:
				images.append(self[i][0])
				captions.append(f"Augmented: {self.image_is_augmented(i)}")
			
		return images, captions


class MpiiDataset(Dataset):
	def __init__(self, dataset_path: str, keys_to_use: List[str] = None, transform=None, is_shuffle=True,
				 is_load_label=True):
		"""If keys_to_use is None(not provided) then the whole dataset will be used."""
		self.dir_path = Path(dataset_path)
		self.is_load_label = is_load_label
		self.transform = transform
		self.keys_to_use = keys_to_use

		if self.keys_to_use == None:
			h5_files = self.dir_path.glob('*.h5')
			self.keys_to_use = [h5_file.name for h5_file in h5_files]
   

		assert len(self.keys_to_use) > 0

		self.image_locations = []

		for key in self.keys_to_use:
			file_path = self.dir_path / key
			assert file_path.is_file()
			h5_file = h5py.File(file_path, 'r', swmr=True)
			for n in range(h5_file["face_patch"].shape[0]):
				self.image_locations.append(
					{
					'h5_file_name': key,
					'dataset_name': 'face_patch',
					'label_dataset_name': 'face_gaze',
					'image_index': (n),
					'label_index': (n),
					}	
				)
			h5_file.close()

		if is_shuffle:
			print('shuffle')
			random.shuffle(self.image_locations)  # random the order to stable the training
  
	def __len__(self):
		return len(self.image_locations)

	def __del__(self):
		pass

	def __getitem__(self, idx):
		image_location =  self.image_locations[idx]
		hdf = h5py.File(self.dir_path / image_location['h5_file_name'], 'r', swmr=True)
		assert hdf.swmr_mode

		# Get face image
		image = hdf[image_location['dataset_name']][image_location['image_index']]
		image = image[:, :, [2, 1, 0]]  # from BGR to RGB
		image = trans(np.uint8(image))
	
		if self.transform:
			image = self.transform(image)
		
		# Get labels
		if self.is_load_label:
			gaze_label = hdf[image_location['label_dataset_name']][image_location['label_index']]
			gaze_label = gaze_label.astype('float')
			return image, gaze_label
		else:

			return image 

	def get_image_samples_and_captions(self, number_of_samples):
		assert number_of_samples < len(self)
    
		sample_index = random.sample(range(len(self)), number_of_samples)
		
		images = []
		captions = []
		if self.is_load_label:
			for i in sample_index:
				images.append(self[i][0])
				captions.append(f"Gaze Direction: {self[i][1]}")
		else: 
			for i in sample_index:
				images.append(self[i][0])
				captions.append(f"Face Patch")
			
		return images, captions

