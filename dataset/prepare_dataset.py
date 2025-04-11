import os
import mrcfile
import math
import random
import pathlib
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
from util.utils import normalize
import random

class CustomDataset(Dataset):
    def __init__(self, mode='train', 
                 base_dir='/storage_data/su_xiaofeng/relion/particle_pick/all_data', 
                 diameter = 24,detect_size=40, padding=12,empty_num=0):
        """ Total cube size = detect_size + 2 * padding
            Make sure total cube size can be divided by 16 (for YOLO)
        """

        self.base_dir = pathlib.Path(base_dir)
        self.mode = mode
        if self.mode == 'train':
            self.data_range = list(self.base_dir.joinpath('train').glob('*.mrc'))
            self.location_range = list(self.base_dir.joinpath('train').glob('*.txt'))
            self.data_range.sort()
            self.location_range.sort()
        elif self.mode == 'val':
            self.data_range = list(self.base_dir.joinpath('eval').glob('*.mrc'))
            self.location_range = list(self.base_dir.joinpath('eval').glob('*.txt'))
            self.data_range.sort()
            self.location_range.sort()
        elif self.mode == 'test':
            self.data_range = list(self.base_dir.joinpath('test').glob('*.mrc'))
            self.location_range = list(self.base_dir.joinpath('test').glob('*.txt'))
            self.data_range.sort()
            self.location_range.sort()
            
        self.diam = diameter
        self.detect_size = detect_size
        self.padding = padding
        self.cube_size = self.detect_size + 2 * self.padding
        self.empty_num = empty_num

        self.origin = []
        self.position = []
        for i in range(len(self.data_range)):
            with mrcfile.open(self.data_range[i],permissive=True) as rec_vol:
                data = rec_vol.data[:, :, :]
                data = normalize(data)
            self.origin.append(data)
            coords = pd.read_csv(self.location_range[i], sep='\t', header=None).to_numpy() 
            self.position.append(coords)

        #set up full volume and label for use
        self.full_volume = []
        for i in range(len(self.data_range)):
            d, h, w = self.origin[i].shape
            vol_w = math.ceil((w - self.padding) / self.detect_size) * self.detect_size + 2 * self.padding
            vol_h = math.ceil((h - self.padding) / self.detect_size) * self.detect_size + 2 * self.padding
            vol_d = math.ceil((d - self.padding) / self.detect_size) * self.detect_size + 2 * self.padding

            volume = np.zeros((vol_d, vol_h, vol_w), dtype=np.float32)
            volume[:d, :h, :w] = self.origin[i]
            self.full_volume.append(volume)
        
        #set up label
        
        #set data and label for training and test
        self.data = []
        self.targets = []
        for i in range(len(self.data_range)):
            base_index = len(self.targets)

            #slices data
            data_vol = self.full_volume[i]
            len_x = (data_vol.shape[2] - 2 * self.padding) // self.detect_size
            len_y = (data_vol.shape[1] - 2 * self.padding) // self.detect_size
            len_z = (data_vol.shape[0] - 2 * self.padding) // self.detect_size

            for z in range(len_z):
                for y in range(len_y):
                    for x in range(len_x):
                        data_cube = data_vol[z * self.detect_size:(z + 1) * self.detect_size + 2 * padding, 
                                             y * self.detect_size:(y + 1) * self.detect_size + 2 * padding, 
                                             x * self.detect_size:(x + 1) * self.detect_size + 2 * padding]
                        data_cube = normalize(data_cube)
                        self.data.append(data_cube)
                        self.targets.append(torch.rand(0, 6))
            
            #make label
            for particle in self.position[i]:
                x, y, z = particle[0:3]
                x = int(round(x))
                y = int(round(y))
                z = int(round(z))
                
                if x < self.padding or y < self.padding or z < self.padding:
                    continue

                t_index = base_index + (x - self.padding) // self.detect_size + (y - self.padding) // self.detect_size * len_x + (z - self.padding) // self.detect_size * len_x * len_y
                self.targets[t_index] = torch.cat((self.targets[t_index], torch.tensor([[(x - self.padding) % self.detect_size + self.padding, (y - self.padding) % self.detect_size + self.padding, (z - self.padding) % self.detect_size + self.padding, self.diam, 1.0, 1.0]])), dim=0)
        
        #clear data
        if self.mode == 'train':
            empty_list = []
            for i in range(len(self.data)):
                if len(self.targets[i]) == 0:
                    empty_list.append(i)
            
            #add empty data
            random.shuffle(empty_list)
            empty_list = empty_list[self.empty_num:]
            empty_list.sort(reverse=True)

            for i in empty_list:
                del self.data[i]
                del self.targets[i]
        
        print('total data: ', len(self.data))
        #print
        print('Setup ', mode, ' dataset ok.')

    def __getitem__(self, index):
        data = self.data[index]
        label = self.targets[index]

        if self.mode == 'train':
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
            label[:, a], label[:, b] = label[:, b], (dim-1) - label[:, a]
        elif degree == 2:
            label[:, a], label[:, b] = (dim-1) - label[:, a], (dim-1) - label[:, b]
        elif degree == 3:
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
    
    def __ContrastNormalizationAugmentation(self,data, label, alpha_range=[0.5, 2.0]):
        if len(alpha_range) != 2 or alpha_range[0] < 0 or alpha_range[1] <= alpha_range[0]:
            raise ValueError(
				"alpha_range must be a list of two values: [min_alpha, max_alpha] where min_alpha < max_alpha.")
        
        rand_multiplier = alpha_range[0] + np.random.rand() * (alpha_range[1] - alpha_range[0])
        middle = np.median(data)
        
        np.subtract(data, middle, out=data)
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
        if max_num == 0: #no object, make sure never occur
            padding_targets = torch.FloatTensor(imgs.size(0), 1, 6).fill_(0)
        else:
            padding_targets = torch.FloatTensor(imgs.size(0), max_num, 6).fill_(0)
            for i, boxes in enumerate(targets):
                padding_targets[i, :boxes.size(0), :] = boxes
        
        return imgs, padding_targets

    def joint(self, pred):
        """
            Must be in eval or test mode, and rotate is Fasle.
        """
        full_pred_list = []
        base_index = 0
        for i in range(len(self.data_range)):
            remove_count = 0
            volume_list = np.zeros((0, 6), dtype=np.int64)
            full_vol = self.full_volume[i]
            rec_vol = self.origin[i]

            len_x = (full_vol.shape[2] - 2 * self.padding) // self.detect_size
            len_y = (full_vol.shape[1] - 2 * self.padding) // self.detect_size
            len_z = (full_vol.shape[0] - 2 * self.padding) // self.detect_size

            for z in range(len_z):
                for y in range(len_y):
                    for x in range(len_x):
                        cube_index = base_index + x + y * len_x + z * len_x * len_y
                        #clear data
                        cube_list = pred[cube_index].numpy().round().astype(np.int64)
                        if len(cube_list) == 0:
                            continue
                        #remove padding
                        x_map = (cube_list[:, 0] >= self.padding) * (cube_list[:, 0] <= self.cube_size - self.padding)
                        y_map = (cube_list[:, 1] >= self.padding) * (cube_list[:, 1] <= self.cube_size - self.padding)
                        z_map = (cube_list[:, 2] >= self.padding) * (cube_list[:, 2] <= self.cube_size - self.padding)
                        cube_list = cube_list[x_map * y_map * z_map]
                        
                        cube_list += np.array([[x * self.detect_size, y * self.detect_size, z * self.detect_size, 0, 0, 0]])
                        #remove out point (important for evaluation)
                        x_map = cube_list[:, 0] < rec_vol.shape[2]
                        y_map = cube_list[:, 1] < rec_vol.shape[1]
                        z_map = cube_list[:, 2] < rec_vol.shape[0]
                        cube_list = cube_list[x_map * y_map * z_map]
                        if cube_list.shape[0] == 0:
                            continue
                        volume_list = np.concatenate((volume_list, cube_list), axis=0)
                        remove_count += len(pred[cube_index]) - len(cube_list)
            
            full_pred_list.append(volume_list)

            if remove_count > 0:
                print('Remove particle: ', remove_count)
            
            return full_pred_list
        
if __name__ == '__main__':

    dataset_dir = '/storage_data/su_xiaofeng/relion/particle_pick'
    train_dataset = CustomDataset(mode = 'train',base_dir=dataset_dir, diameter=28,)
    train_data = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=8,
		shuffle=True,
		num_workers=4,
		collate_fn=train_dataset.collate_fn
	)
    
    for batch_i, (imgs, targets) in enumerate(train_data):
            imgs = imgs
            targets = targets
            print(imgs.shape)
            print(targets.shape)

    print('-----------')