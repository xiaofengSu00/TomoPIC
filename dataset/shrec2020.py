import os
import math
import mrcfile
import warnings
import numpy as np
import pandas as pd
from util.utils import normalize
import torch
import random
from torch.utils.data.dataset import Dataset
from scipy.ndimage.interpolation import map_coordinates


class SHREC2020(Dataset):
	"""SHREC2020 Dataset for 3D detection work.
	   Cube size: 64 x 64 x 64, slicing from 192x512x512 volume, which deprecates first 4 layers and last 4 layers on z-coordinate.

	   collate_fn: padding targets of different length to same length for batching
	"""

	def __init__(self, mode='train',
				 base_dir='/storage_data/su_xiaofeng/shrec_data/shrec_2020/shrec2020_full_dataset',
				 detect_size=48, padding=8, aug_choice=0):
		""" Total cube size = detect_size + 2 * padding
			Make sure total cube size can be divided by 16 (for YOLO)
		"""

		self.base_dir = base_dir
		self.mode = mode
		if self.mode == 'train':
			self.data_range = [0, 1, 2, 3, 4, 5, 6, 7]
		elif self.mode == 'val':
			self.data_range = [8]
		else:  # test
			self.data_range = [9]

		self.detect_size = detect_size
		self.padding = padding
		self.cube_size = self.detect_size + 2 * self.padding
		self.aug_choice = aug_choice

		self.num_class = 12  # 1 or 12

		# dictionary to get particle class and size by name, no class 0
		
		# self.particle_dict = {'3cf3': [1, 14],
        #                       '1s3x': [2, 6],
        #                       '1u6g': [3, 13],
        #                       '4cr2': [4, 24],
        #                       '1qvr': [5, 17],
        #                       '3h84': [6, 14],
        #                       '2cg9': [7, 12],
        #                       '3qm1': [8, 7],
        #                       '3gl1': [9, 7],
        #                       '3d2f': [10, 13],
        #                       '4d8q': [11, 16],
        #                       '1bxn': [12, 12]}
		self.particle_dict = {'3cf3': [1, 13],
                              '1s3x': [2, 6],
                              '1u6g': [3, 12],
                              '4cr2': [4, 16],
                              '1qvr': [5, 10],
                              '3h84': [6, 9],
                              '2cg9': [7, 10],
                              '3qm1': [8, 6],
                              '3gl1': [9, 7],
                              '3d2f': [10, 12],
                              '4d8q': [11, 12],
                              '1bxn': [12, 11]}
		# raw data
		self.reconstruction_volume = []
		self.location = []

		# read data and location files
		warnings.simplefilter('ignore')
		for i in self.data_range:
			# with mrcfile.open(os.path.join(self.base_dir, 'model_%d/grandmodel_noisefree.mrc' % i), permissive=True) as rec_vol:
			with mrcfile.open(os.path.join(self.base_dir, 'model_%d/reconstruction.mrc' % i),
							  permissive=True) as rec_vol:
				data = rec_vol.data[156:-156, :, :]
				data = normalize(data)
				self.reconstruction_volume.append(data)
			self.location.append(
				pd.read_csv(os.path.join(self.base_dir, 'model_%d/particle_locations.txt' % i), header=None, sep=' '))

		# set up full volume and label for use
		self.full_volume = []
		for i in range(len(self.data_range)):
			d, h, w = self.reconstruction_volume[i].data.shape
			vol_w = math.ceil((w - self.padding) / self.detect_size) * self.detect_size + 2 * self.padding
			vol_h = math.ceil((h - self.padding) / self.detect_size) * self.detect_size + 2 * self.padding
			vol_d = math.ceil((d - self.padding) / self.detect_size) * self.detect_size + 2 * self.padding



			volume = np.zeros((vol_d, vol_h, vol_w), dtype=np.float32)
			volume[:d, :h, :w] = self.reconstruction_volume[i].data
			self.full_volume.append(volume)

		# data and label for training and test
		self.data = []
		self.label = []
		for i in range(len(self.data_range)):
			base_index = len(self.label)
			rec_vol = self.full_volume[i]

			len_x = (rec_vol.shape[2] - 2 * self.padding) // self.detect_size
			len_y = (rec_vol.shape[1] - 2 * self.padding) // self.detect_size
			len_z = (rec_vol.shape[0] - 2 * self.padding) // self.detect_size

			for z in range(len_z):  # 11
				for y in range(len_y):  # 11
					for x in range(len_x):  # 11
						data_cube = rec_vol[z * self.detect_size:(z + 1) * self.detect_size + 2 * self.padding,
									y * self.detect_size:(y + 1) * self.detect_size + 2 * self.padding,
									x * self.detect_size:(x + 1) * self.detect_size + 2 * self.padding]
						data_cube = normalize(data_cube)
						self.data.append(data_cube)
						self.label.append(torch.rand(0, 5 + self.num_class))

			location = self.location[i]
			for j in range(len(location)):
				particle = location.loc[j]

				if particle[0] not in self.particle_dict.keys():
					continue

				# print(particle[0])

				p_class, p_size = self.particle_dict[particle[0]]
				x, y, z = particle[1:4]

				if x < self.padding or y < self.padding or z < self.padding:
					continue

				label_index = base_index + (x - self.padding) // self.detect_size + (
							y - self.padding) // self.detect_size * len_x + (
										  z - self.padding) // self.detect_size * len_x * len_y
				if self.num_class == 1:
					new_item = torch.FloatTensor([[(x - self.padding) % self.detect_size + self.padding,
												   (y - self.padding) % self.detect_size + self.padding,
												   (z - self.padding) % self.detect_size + self.padding, p_size, 1, 1]])
				else:
					new_item = torch.FloatTensor([[(x - self.padding) % self.detect_size + self.padding,
												   (y - self.padding) % self.detect_size + self.padding,
												   (z - self.padding) % self.detect_size + self.padding, p_size, 1, 0,
												   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
					new_item[0, 4 + p_class] = 1
				self.label[label_index] = torch.cat((self.label[label_index], new_item), dim=0)

				# print
		print('Setup ', mode, ' dataset ok.')

	def __getitem__(self, index):
		data = self.data[index]
		label = self.label[index]

		if self.aug_choice == 1:
			data, label = self.__rotation3D(data, label)

		elif self.aug_choice == 2:
			data, label = self.__translate3D(data, label)
				
		elif self.aug_choice == 3:
			data,label = self.__ContrastNormalizationAugmentation(data,label)
			
		elif self.aug_choice == 4:
			choice = np.random.rand()
			if choice <= 0.3:
				data,label = self.__ContrastNormalizationAugmentation(data,label)
				data, label = self.__rotation3D(data, label)
			elif 0.3 < choice <= 0.6:
				data,label = self.__ContrastNormalizationAugmentation(data,label)
				data, label = self.__translate3D(data, label)
			else:
				data,label = self.__ContrastNormalizationAugmentation(data,label)
				data, label = self.__rotation3D(data, label)
				data, label = self.__translate3D(data, label)
		

		data = torch.tensor(np.array(data)).unsqueeze(0)

		return data, label

	def __len__(self):
		return len(self.data)

	def __rotate_label(self, label, degree, axis,dim):
		# axis是一个二元组，例如(0, 1)表示在前两个维度旋转
		a, b = axis
		a = 2 - a
		b = 2 - b

		if degree == 1:
			# 旋转90°：label[:, a], label[:, b] = label[:, b], 64 - label[:, a]
			label[:, a], label[:, b] = label[:, b], (dim-1) - label[:, a]
		elif degree == 2:
			# 旋转180°：label[:, a], label[:, b] = 64 - label[:, a], 64 - label[:, b]
			label[:, a], label[:, b] = (dim-1) - label[:, a], (dim-1) - label[:, b]
		elif degree == 3:
			# 旋转270°：label[:, a], label[:, b] = 64 - label[:, b], label[:, a]
			label[:, a], label[:, b] = (dim-1) - label[:, b], label[:, a]

		return label

	def __rotation3D(self, data, label):

		choice = random.randint(0, 2)
		degree = random.randint(1, 3)  # 随机选择旋转角度: 90°, 180°, 270°

		if choice == 0:
			data = np.rot90(data, degree, axes=(0, 1))
			label = self.__rotate_label(label, degree, (0, 1),dim=data.shape[0])
		elif choice == 1:
			data = np.rot90(data, degree, axes=(1, 2))
			label = self.__rotate_label(label, degree, (1, 2),dim=data.shape[0])
		elif choice == 2:
			data = np.rot90(data, degree, axes=(0, 2))
			label = self.__rotate_label(label, degree, (0, 2),dim=data.shape[0])

		return data, label

	def __translate3D(self, data, label):
		z_shift = random.randint(-2, 2)
		y_shift = random.randint(-2, 2)
		x_shift = random.randint(-2, 2)

		z_dim, y_dim, x_dim = data.shape

		# Create an empty array with the same shape
		translated_data = np.zeros_like(data)

		# Calculate valid slicing ranges
		z_range = slice(max(0, -z_shift), min(z_dim, z_dim - z_shift))
		y_range = slice(max(0, -y_shift), min(y_dim, y_dim - y_shift))
		x_range = slice(max(0, -x_shift), min(x_dim, x_dim - x_shift))

		# Perform translation
		translated_data[z_range, y_range, x_range] = data[
			slice(max(0, z_shift), min(z_dim, z_dim + z_shift)),
			slice(max(0, y_shift), min(y_dim, y_dim + y_shift)),
			slice(max(0, x_shift), min(x_dim, x_dim + x_shift))]

		label[:, 2] = np.clip(label[:, 2] + z_shift, 0, z_dim - 1)
		label[:, 1] = np.clip(label[:, 1] + y_shift, 0, y_dim - 1)
		label[:, 0] = np.clip(label[:, 0] + x_shift, 0, x_dim - 1)

		return translated_data, label

	def __flip3D(self, data, label):
		axes = [random.randint(0, 2)]

		if not isinstance(axes, list):
			raise ValueError("Axes should be a list of integers representing the axes to flip along.")
		# Validate axes
		for axis in axes:
			if axis not in [0, 1, 2]:
				raise ValueError("Invalid axis. Choose 0 for z, 1 for y, or 2 for x.")

		# Map axes to original order (zyx to xyz)
		axis_map = {0: 2, 1: 1, 2: 0}

		for axis in axes:
			data = np.flip(data, axis=axis_map[axis])

		label = self.__fliplabel(data, label, axes)

		return data, label

	def __fliplabel(self, data, label, flip_axes):
		# z, y, x = label[:,2], label[:, 1], label[:, 0]
		z_dim, y_dim, x_dim = data.shape

		# Adjust coordinates based on flipped axes
		if 0 in flip_axes:  # z axis flipped
			label[:, 2] = z_dim - label[:, 2] - 1
		if 1 in flip_axes:  # y axis flipped
			label[:, 1] = y_dim - label[:, 1] - 1
		if 2 in flip_axes:  # x axis flipped
			label[:, 0] = x_dim - label[:, 0] - 1

		return label
	def __ContrastNormalizationAugmentation(self,data, label, alpha_range=[0.5, 2.0]):
		if len(alpha_range) != 2 or alpha_range[0] < 0 or alpha_range[1] <= alpha_range[0]:
			raise ValueError(
				"alpha_range must be a list of two values: [min_alpha, max_alpha] where min_alpha < max_alpha.")
		# data = np.asarray(data)

		# Determine the alpha multiplier
		rand_multiplier = alpha_range[0] + np.random.rand() * (alpha_range[1] - alpha_range[0])

		# Calculate the median across all elements in the image
		middle = np.median(data)

		# Normalize the pixel values
		# Subtract the median
		np.subtract(data, middle, out=data)

		# Multiply by the random alpha multiplier
		np.multiply(rand_multiplier, data, out=data)

		# Add the median back
		np.add(middle, data, out=data)

		return data, label

	def collate_fn(self, batch):
		"""
			Padding targets to same size for batching
		"""
		imgs, targets = list(zip(*batch))
		imgs = torch.stack([img for img in imgs])

		max_num = max([boxes.size(0) for boxes in targets])
		if max_num == 0:  # no object, make sure never occur
			padding_targets = torch.FloatTensor(imgs.size(0), 1, 5 + self.num_class).fill_(0)
		else:
			padding_targets = torch.FloatTensor(imgs.size(0), max_num, 5 + self.num_class).fill_(0)
			for i, boxes in enumerate(targets):
				padding_targets[i, :boxes.size(0), :] = boxes

		return imgs, padding_targets

	def joint(self, pred):  ### pred---- list[tensor(pred_bbox,class_num+5)]
		"""
			Restore the total list of the full volume.
			Must be in eval or test mode, and rotate is Fasle.

			iterate pred_list and then
		"""
		full_pred_list = []
		base_index = 0
		for i in range(len(self.data_range)):
			remove_count = 0
			volume_list = np.zeros((0, 6), dtype=np.int16)
			full_vol = self.full_volume[i]
			rec_vol = self.reconstruction_volume[i]

			len_x = (full_vol.shape[2] - 2 * self.padding) // self.detect_size
			len_y = (full_vol.shape[1] - 2 * self.padding) // self.detect_size
			len_z = (full_vol.shape[0] - 2 * self.padding) // self.detect_size

			for z in range(len_z):  # 4
				for y in range(len_y):  # 11
					for x in range(len_x):  # 11
						cube_index = base_index + x + y * len_x + z * len_x * len_y
						# clear data
						cube_list = pred[cube_index].numpy()  ### 每个cube的经NMS后的预测结果
						cube_list[:, 4] = cube_list[:, 4] * 1000
						cube_list = cube_list.round().astype(np.int16)

						if self.num_class > 1:
							# re-write class number
							cube_list[:, 5] = np.argmax(cube_list[:, 5:], axis=1) + 1  # 1-12
							cube_list = cube_list[:, :6]

						# remove padding
						x_map = (cube_list[:, 0] >= self.padding) * (cube_list[:, 0] <= self.cube_size - self.padding)
						y_map = (cube_list[:, 1] >= self.padding) * (cube_list[:, 1] <= self.cube_size - self.padding)
						z_map = (cube_list[:, 2] >= self.padding) * (cube_list[:, 2] <= self.cube_size - self.padding)
						cube_list = cube_list[x_map * y_map * z_map]  ### 筛选出 xyz 大于 padding的预测结果

						# cube_list += np.array([[x * self.detect_size - self.padding,
						# 						y * self.detect_size - self.padding,
						# 						z * self.detect_size - self.padding,
						# 						  0, 0, 0]])
						cube_list += np.array([[x * self.detect_size ,
												y * self.detect_size ,
												z * self.detect_size ,
												  0, 0, 0]])

						# remove out point (important for evaluation)
						x_map = cube_list[:, 0] < rec_vol.shape[2]
						y_map = cube_list[:, 1] < rec_vol.shape[1]
						z_map = cube_list[:, 2] < rec_vol.shape[0]
						cube_list = cube_list[x_map * y_map * z_map]
						if cube_list.shape[0] == 0:
							continue
						volume_list = np.concatenate((volume_list, cube_list), axis=0)
						remove_count += len(pred[cube_index]) - len(cube_list)

			full_pred_list.append(volume_list)

			# if remove_count > 0:
			#    print('Remove particle: ', remove_count)

			return full_pred_list



if __name__ == '__main__':
	dataset = SHREC2020(mode='train',
						base_dir='/storage_data/su_xiaofeng/shrec_data/shrec_2020/shrec2020_full_dataset',
						detect_size=48, aug_choice=1)                   