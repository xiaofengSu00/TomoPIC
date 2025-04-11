import argparse
import torch
import numpy as np
from tqdm import tqdm
import time
from util.utils import *
from eval.evaluate import *
from dataset.shrec2020 import SHREC2020
from dataset.Dataset_10045 import Dataset_10045
from model.yolo import Detect_Framework
from model.loss import ComputeLoss
from util.calculate import *
import mrcfile
import pandas as pd
from scipy.spatial.distance import cdist

def evaluate(pred_list,occupancy_file,coords_file):
	
	# occupancy_file = './TS_026_cyto_ribosomes.mrc'
	occupancy = mrcfile.open(occupancy_file).data
	# coords_file = './TS_026_cyto_ribosomes.csv'
	coords = pd.read_csv(coords_file, sep='\t', header=None).to_numpy()
	true_value = []
	if len(pred_list.shape) == 1:
		pred_list = pred_list.reshape(1,-1)
	for i in range(len(pred_list)):
		pred = pred_list[i, :3]
		x, y, z = pred
		if occupancy[z, y, x] != 0:
			true_value.append([x, y, z])
	
	if len(true_value) == 0:
		print("No true values found, returning...")
		print('TP: 0, FP: 0, FN: 0 \t Precision: 0, Recall: 0\n' )
		return 0
	true_value = np.array(true_value)
	distance_matrix = cdist(true_value,coords,metric='euclidean')
	min_indices = np.argmin(distance_matrix, axis=1)
	TP = len(np.unique(min_indices))
	FN = len(coords) - TP
	FP = len(true_value) - TP
	precision = TP / (TP + FP + 1e-6)
	recall = TP / (TP + FN + 1e-6)
	print('TP: %d, FP: %d, FN: %d \t Precision: %.6f, Recall: %.6f\n' % (
	TP, FP, FN, TP / (TP + FP + 1e-6), TP / (TP + FN + 1e-6)))
	return 2 * (precision * recall) / (precision + recall + 1e-6)

def test(opt):
   
    ##########preparing
    device = prepare_devices(opt)

    if opt.choice == 'shrec2020':
        eval_dataset = SHREC2020(mode='test', base_dir=opt.dataset_dir)
    
    elif opt.choice == 'shrec2021':
        eval_dataset = SHREC2020(mode='test', base_dir=opt.dataset_dir)
    else:
        eval_dataset = Dataset_10045(mode='test', base_dir=opt.dataset_dir)
    
    eval_data = torch.utils.data.DataLoader(
        eval_dataset, 
        batch_size=opt.batch_size, 
        shuffle=False, 
        num_workers=opt.num_workers, 
        collate_fn=eval_dataset.collate_fn
    )

    """
        Pre-cluster results, 9 types
    """

    if opt.choice == 'shrec2020':
        ANCHOR = torch.tensor([[[11.4], [17.7], [29]]])

        model = Detect_Framework(class_num=12, anchors=ANCHOR, anchor_num=3)
        model = load_model(model, opt)

        compute_loss = ComputeLoss(model = model,
                class_num=12, 
                anchor_num=3, 
                anchor= ANCHOR)
        
    elif opt.choice == 'shrec2021':
        ANCHOR = torch.tensor([[[13.5], [20.3], [34.5]]])
        model = Detect_Framework( class_num=13, anchors=ANCHOR,anchor_num=3)
        model = load_model(model, opt)

        compute_loss = ComputeLoss(model = model,
                class_num=13, 
                anchor_num=3, 
                anchor= ANCHOR)
    else:
        ANCHOR = torch.tensor([[[24]]])
        model = Detect_Framework( class_num=1, anchors=ANCHOR,anchor_num=1)
        model = load_model(model, opt)

        compute_loss = ComputeLoss(model = model,
                    class_num=1, 
                    anchor_num=1, 
                    anchor= ANCHOR)

    print("---- Evaluating Model ----")
    model.eval()

    ##########
    conf_thres = 0.5
    nms_thres = 0.1
    ##########
    loss_list = []
    loss_conf_list = []
    loss_cls_list = []

    start_time = time.time()

    with torch.no_grad():
        pred_list = []
        for batch_i, (imgs, targets) in enumerate(tqdm(eval_data, desc=f"Calculating")):
            imgs = imgs.to(device)
            targets = targets.to(device)
            train_out,pred = model(imgs)
            loss, loss_conf, loss_cls = compute_loss(train_out,targets)

            loss = loss.mean()
            loss_conf = loss_conf.mean()
            loss_cls = loss_cls.mean()
            loss_list.append(loss.cpu().detach())
            loss_conf_list.append(loss_conf.cpu().detach())
            loss_cls_list.append(loss_cls.cpu().detach())

            for b in range(pred.size(0)):
                mask = pred[b, ..., 4] > conf_thres

                b_pred = pred[b,mask,:]
                b_pred = b_pred.cpu()
                index = Non_Maximum_Suppression(b_pred[:, :5], nms_thres)
                b_pred = b_pred[index, :]
                pred_list.append(b_pred)

        result = eval_dataset.joint(pred_list)

        total_pred_list = result[0]
        final_pred_list = total_pred_list[remove(torch.from_numpy(total_pred_list[:, :5].astype(np.float32))), :]
        
        end_time = time.time()
        epoch_duration = end_time - start_time
        print('Test Time:{:.2f}'.format(epoch_duration))
        print('EVAL Result:')
        print('Loss: %.6f \t Conf Loss: %.6f \t Class Loss: %.6f' % (np.mean(loss_list), np.mean(loss_conf_list), np.mean(loss_cls_list)))
        # print(final_pred_list[1:80,:])
        
        if opt.choice == 'shrec2020':
            fscore = evaluate_shrec2020(final_pred_list,base_dir=opt.dataset_dir,data_id=9)
        elif opt.choice == 'shrec2021':
            fscore = evaluate_shrec2021(final_pred_list,base_dir=opt.dataset_dir,data_id=9)
        else:
            fscore = evaluate(final_pred_list,occupancy_file='/storage_data/su_xiaofeng/relion/particle_pick/all_data/tomo_mask/IS002_291013_011.mrc',
                                        coords_file='/storage_data/su_xiaofeng/relion/particle_pick/all_data/coords/IS002_291013_011.txt')
				
        print(fscore)

        df = pd.DataFrame(final_pred_list)
        df.to_csv('./test_result/10045/reslut_18.csv',sep=',',index=False)
        
def create_parser():
    parser = argparse.ArgumentParser()
    #project options
    parser.add_argument('--gpu_ids', type=str, default='0', help='GPU ids, use -1 for CPU.')
    parser.add_argument('--load_dir', type=str, default='./checkpoints_10045/2024-11-14-20:51:48', help='The directory of the pretrained model.')
    """important"""
    parser.add_argument('--dataset_dir', type=str, default='/storage_data/su_xiaofeng/relion/particle_pick/all_data', help='The directory of the used dataset')

    #testing options
    parser.add_argument('--load_filename', type=str, default='best_network.pth', help='Filename of the pretrained model.')

    # choice
    parser.add_argument('--choice',type=str,default='10045',help='shrec2020,shrec2020,10045')
    #dataset options
    parser.add_argument("--batch_size", type=int, default=12, help="Size of each image batch.")
    parser.add_argument("--num_workers", type=int, default=4, help="number of cpu threads to use during batch generation")
    opt = parser.parse_args()

    return opt

if __name__ =='__main__':
    opt = create_parser()
    test(opt)

        