import os
import argparse
import mrcfile
import math
import pathlib
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from util.utils import *
from model.yolo import Detect_Framework
from util.calculate import *
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

class PredictDataset(Dataset):
    def __init__(self, tomo_path,detect_size=40, padding=12):
        """ Total cube size = detect_size + 2 * padding
            Make sure total cube size can be divided by 16 (for YOLO)
        """
        self.tomo = pathlib.Path(tomo_path)
        self.detect_size = detect_size
        self.padding = padding
        self.cube_size = self.detect_size + 2 * self.padding
    
        with mrcfile.open(self.tomo,permissive=True) as rec_vol:
            data = rec_vol.data[:, :, :]
            self.reconstruction_volume = normalize(data)
            
            
        #set up full volume and label for use
        d, h, w = self.reconstruction_volume.shape
        vol_w = math.ceil((w - self.padding) / self.detect_size) * self.detect_size + 2 * self.padding
        vol_h = math.ceil((h - self.padding) / self.detect_size) * self.detect_size + 2 * self.padding
        vol_d = math.ceil((d - self.padding) / self.detect_size) * self.detect_size + 2 * self.padding

        self.full_volume = np.zeros((vol_d, vol_h, vol_w), dtype=np.float32)
        self.full_volume[:d, :h, :w] = self.reconstruction_volume
        
        
        #set up label
        
        #set data for prediction
        self.data = []
        len_x = (self.full_volume.shape[2] - 2 * self.padding) // self.detect_size
        len_y = (self.full_volume.shape[1] - 2 * self.padding) // self.detect_size
        len_z = (self.full_volume.shape[0] - 2 * self.padding) // self.detect_size

        for z in range(len_z):
            for y in range(len_y):
                for x in range(len_x):
                    data_cube = self.full_volume[z * self.detect_size:(z + 1) * self.detect_size + 2 * padding, 
                                             y * self.detect_size:(y + 1) * self.detect_size + 2 * padding, 
                                             x * self.detect_size:(x + 1) * self.detect_size + 2 * padding]
                    self.data.append(data_cube)
    
    def __getitem__(self, index):
        data = self.data[index]
        data = torch.tensor(np.array(data)).unsqueeze(0)
        return data
    
    def __len__(self):
        return len(self.data)
    
    def joint(self, pred):
        """
            Must be in eval or test mode, and rotate is Fasle.
        """
        full_pred_list = []
        base_index = 0
       
        remove_count = 0
        volume_list = np.zeros((0, 6), dtype=np.int64)
        full_vol = self.full_volume
        rec_vol = self.reconstruction_volume

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


def create_parser():
    parser = argparse.ArgumentParser()
    # project options
    parser.add_argument('--model_name', type=str, default='YOLO3D', help='Name of this experiment.')
    parser.add_argument('--gpu_ids', type=str, default='0', help='GPU ids, use -1 for CPU.')
    
    parser.add_argument('--load_dir', type=str, default='./checkpoints_10045/0220_epoch500', help='The directory of the pretrained model.')

    parser.add_argument('--dataset_dir', type=str,
						default='/storage_data/su_xiaofeng/data/other_method/ETyolopicker/test',
						help='The directory of the used dataset')
    ## dataset options
    parser.add_argument('--num_class', type=int, default=1, help='the number of class')
    parser.add_argument("--batch_size", type=int, default=10, help="Size of each image batch.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of cpu threads to use during batch generation")

    ## model options
    parser.add_argument('--anchor',type=int,nargs='+',default=24,help = 'the particle diameter for one class or the pre-cluster result for mulitiple class')

    # predict options
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained model.')

    parser.add_argument('--load_filename', type=str, default='best_network.pth',
						help='Filename of the pretrained model.')

    opt = parser.parse_args()

    return opt

def predict(opt):
    device = prepare_devices(opt)

    opt.dataset_dir = pathlib.Path(opt.dataset_dir)
    
    if opt.dataset_dir.is_dir():
        data_range = list(opt.dataset_dir.glob('*.mrc'))
        data_range.sort()
    
    import time
    
    
    if isinstance(opt.anchor,int):
        ANCHOR = torch.tensor([opt.anchor]).resize(1,1,1)
        model = Detect_Framework(class_num=opt.num_class, anchors=ANCHOR, anchor_num=1)
    elif isinstance(opt.anchor,list):
        ANCHOR = torch.tensor(opt.anchor).resize(1,len(opt.anchor),1)
        model = Detect_Framework(class_num=opt.num_class, anchors=ANCHOR, anchor_num=len(opt.anchor))

    model = load_model(model, opt)
    start_time = time.time()
    for tomo in data_range:
        data_name = tomo.name.split('.')[0]
        pred_dataset = PredictDataset(tomo_path=tomo,detect_size=40,padding=12)
        pred_data = torch.utils.data.DataLoader(
            pred_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=4,
        )
      
        model.eval()
        ##########
        conf_thres = 0.5
        nms_thres = 0.1

        with torch.no_grad():
            pred_list = []
            for batch_i,imgs in enumerate(tqdm(pred_data, desc=f"Calculating")):
                imgs = imgs.to(device)
                
                _,pred = model(imgs)

                for b in range(pred.size(0)):
                    mask = pred[b, ..., 4] > conf_thres

                    b_pred = pred[b,mask,:]
                    b_pred = b_pred.cpu()
                    index = Non_Maximum_Suppression(b_pred[:, :5], nms_thres)
                    b_pred = b_pred[index, :]
                    pred_list.append(b_pred)
            
            result = pred_dataset.joint(pred_list)

            total_pred_list = result[0]
            final_pred_list = total_pred_list[remove(torch.from_numpy(total_pred_list[:, :5].astype(np.float32))), :]
            
            
            df = pd.DataFrame(final_pred_list[:,:3])
            save_file = './test_result/10045/0220_epoch340/{}.txt'.format(data_name)
            if os.path.exists(save_file):
                os.remove(save_file)
            df.to_csv(save_file,header=False,index=False,sep='\t')

    end_time = time.time()
    epoch_duration = end_time - start_time
    print('Prediction Time:{:.2f}'.format(epoch_duration))

if __name__ == '__main__':
    
    opt = create_parser()

    predict(opt)