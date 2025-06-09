import os
import mrcfile
import warnings
import numpy as np
import pandas as pd
from skimage.morphology import dilation


def eval(pred_list, gt_list, loc2id, class_id=0):
	""" Evaluation without class(one class).
		All data here is ndarray.

		Input:
			pred_list: List of particles predicted by network. Like: [[x, y, z]]
			gt_list: Ground truth location.
			loc2id: Cube to relate predicted location and ground truth point id.
			class_id: 1-12 for one class evaluation. 0 for all.
	"""
	if class_id > 0:
		
		mask = pred_list[:, 5] == class_id
		pred_list_of_one_class = pred_list[mask]
	else:
		pred_list_of_one_class = pred_list

	pred_x = pred_list_of_one_class[:, 0]
	pred_y = pred_list_of_one_class[:, 1]
	pred_z = pred_list_of_one_class[:, 2]

	result = loc2id[pred_z, pred_y, pred_x].astype(np.int64)
	result = np.unique(result)

	gt_class = gt_list[result][:, 0]

	if class_id > 0:
		gt_box = ((gt_list[:, 0] == class_id).astype(np.int16) - 1) * 2  # positive box with 0 and negative box with -2
	else:
		gt_box = np.zeros(gt_list.shape[0])
	gt_box[result] += 1

	FP = sum(gt_class == 0) if class_id == 0 else sum(gt_class != class_id) ### gt_class
	FN = sum(gt_box[1:] == 0)
	TP = sum(gt_box[1:] > 0)
	dp = pred_list_of_one_class.shape[0] - FP - TP  # duplicate
	FP += dp

	return TP, FP, FN, dp


def multi_class_eval(pred_list, gt_list, loc2id, class_num):
	""" Multi-Class Evaluation.
		All data here is ndarray.

		Input:
			pred_list: List of particles predicted by network. Like: [[x, y, z]]
			gt_list: Ground truth location.
			loc2id: Cube to relate predicted location and ground truth point id.
	"""
	PRECISION = np.zeros(class_num, dtype=np.float32)
	RECALL = np.zeros(class_num, dtype=np.float32)
	for i in range(class_num):
		TP, FP, FN, _ = eval(pred_list, gt_list, loc2id, class_id=i + 1)
		PRECISION[i] = TP / (TP + FP + 1e-6)
		RECALL[i] = TP / (TP + FN + 1e-6)

	pred_x = pred_list[:, 0]
	pred_y = pred_list[:, 1]
	pred_z = pred_list[:, 2]
	pred_class = pred_list[:, 5]
	result = loc2id[pred_z, pred_y, pred_x].astype(np.int64)
	gt_class = gt_list[result][:, 0]

	result_map = np.zeros((class_num, class_num + 1), dtype=np.int64)
	for i in range(len(result)):
		result_map[pred_class[i] - 1, gt_class[i]] += 1

	print(result_map)
	print('precision: ', PRECISION)
	print('recall: ', RECALL)
	print('F1 score: ', 2 * PRECISION * RECALL / (PRECISION + RECALL + 1e-6))


def evaluate_shrec2020(pred_list,
					   base_dir='/storage_data/su_xiaofeng/shrec_data/shrec_2020/shrec2020_full_dataset',
					   data_id=8, class_num=12):
	"""
		data_id: 8 for eval and 9 for test.
	"""
	# pred_list =pred_list.reshape
	location = pd.read_csv(os.path.join(base_dir, 'model_%d/particle_locations.txt' % data_id), header=None, sep=' ')
	particle_dict = {'3cf3': 1, '1s3x': 2, '1u6g': 3, '4cr2': 4, '1qvr': 5, '3h84': 6, '2cg9': 7, '3qm1': 8, '3gl1': 9,
					 '3d2f': 10, '4d8q': 11, '1bxn': 12}
	for i in range(len(location)):
		location.loc[i, 0] = particle_dict[location.loc[i, 0]]
	particle_list = np.array(location.loc[:, 0:3])
	gt_list = np.concatenate((np.array([[0, 0, 0, 0]]), particle_list), axis=0)

	warnings.simplefilter('ignore')
	occupancy_map = None
	with mrcfile.open(os.path.join(base_dir, 'model_%d/occupancy_mask.mrc' % data_id), permissive=True) as m:
		occupancy_map = dilation(m.data)

	if class_num:
		multi_class_eval(pred_list, gt_list, occupancy_map, class_num=class_num)

	TP, FP, FN, DP = eval(pred_list, gt_list, occupancy_map)

	precision = TP / (TP + FP + 1e-6)
	recall = TP / (TP + FN + 1e-6)

	print('TP: %d, FP: %d, FN: %d, DP: %d \t Precision: %.6f, Recall: %.6f\n' % (
	TP, FP, FN, DP, TP / (TP + FP + 1e-6), TP / (TP + FN + 1e-6)))

	return 2 * (precision * recall) / (precision + recall + 1e-6)


def evaluate_shrec2021(pred_list,
					   base_dir='/storage_data/su_xiaofeng/shrec_data/shrec_2021',
					   data_id=8, class_num=13):
	"""
		data_id: 8 for eval and 9 for test.
	"""
	pred_list = np.array(pred_list).reshape(-1,6)
	location = pd.read_csv(os.path.join(base_dir, 'model_%d/particle_locations.txt' % data_id), header=None, sep=' ')

	particle_dict = {
					 '4CR2': 1,
					 '1QVR': 2,
					 '1BXN': 3,
					 '3CF3': 4,
					 '1U6G': 5,
					 '3D2F': 6,
					 '2CG9': 7,
					 '3H84': 8,
					 '3GL1': 9,
					 '3QM1': 10,
					 '1S3X': 11,
					 '5MRC': 12,
					 'fiducial': 13}

	for i in range(len(location)):
		location.loc[i, 0] = particle_dict.get(location.loc[i, 0],0)

	particle_list = np.array(location.loc[:, 0:3])
	gt_list = np.concatenate((np.array([[0, 0, 0, 0]]), particle_list), axis=0)

	warnings.simplefilter('ignore')
	occupancy_map = None
	with mrcfile.open(os.path.join(base_dir, 'model_%d/occupancy_mask.mrc' % data_id), permissive=True) as m:
		occupancy_map = dilation(m.data)

	if class_num > 1:
		multi_class_eval(pred_list, gt_list, occupancy_map, class_num=class_num)

	TP, FP, FN, DP = eval(pred_list, gt_list, occupancy_map)

	precision = TP / (TP + FP + 1e-6)
	recall = TP / (TP + FN + 1e-6)

	print('TP: %d, FP: %d, FN: %d, DP: %d \t Precision: %.6f, Recall: %.6f\n' % (
	TP, FP, FN, DP, TP / (TP + FP + 1e-6), TP / (TP + FN + 1e-6)))

	return 2 * (precision * recall) / (precision + recall + 1e-6)



def evaluate_10045(pred_list,coords_file,
				   base_dir='',
				   ):
	pred_list = np.array(pred_list).reshape(-1,6)

	# if len(pred_list) == 1:
	# 	pred_list = np.array(pred_list).reshape(1, -1)
	occupancy_map = mrcfile.open(base_dir).data
	occupancy_map = dilation(occupancy_map)
	
	particle_list = pd.read_csv(coords_file, sep='\t', header=None).to_numpy()
	gt_list = np.concatenate((np.array([[0, 0, 0]]), particle_list), axis=0)

	TP, FP, FN, DP = eval(pred_list=pred_list,gt_list=gt_list,loc2id=occupancy_map)

	precision = TP / (TP + FP + 1e-6)
	recall = TP / (TP + FN + 1e-6)

	print('TP: %d, FP: %d, FN: %d, DP: %d \t Precision: %.6f, Recall: %.6f\n' % (
	TP, FP, FN, DP, TP / (TP + FP + 1e-6), TP / (TP + FN + 1e-6)))

	return 2 * (precision * recall) / (precision + recall + 1e-6)










# if __name__ == "__main__":
# 	import torch
# 	import numpy as np
# 	# base_dir = '/storage_data/su_xiaofeng/shrec_data/shrec_2021'
# 	# data_id = 0
# 	# location = pd.read_csv(os.path.join(base_dir, 'model_%d/particle_locations.txt' % data_id), header=None, sep=' ')
	
# 	# particle_dict = {'4CR2': [1, 33],
#     #                           '1QVR': [2, 25],
#     #                           '1BXN': [3, 18],
#     #                           '3CF3': [4, 22],
#     #                           '1U6G': [5, 16],
#     #                           '3D2F': [6, 21],
#     #                           '2CG9': [7, 18],
#     #                           '3H84': [8, 18],
#     #                           '3GL1': [9, 14],
#     #                           '3QM1': [10, 12],
#     #                           '1S3X': [11, 12],
#     #                           '5MRC': [12, 36],
#     #                           'fiducial': [13,12]}
# 	# label = np.zeros(shape=(0,6),dtype=np.int16)
# 	# for j in range(len(location)):
# 	# 	particle = location.loc[j]
# 	# 	if particle[0] not in particle_dict.keys():
# 	# 		continue
# 	# 	# print(particle[0])
# 	# 	p_class, p_size = particle_dict.get(particle[0],[0,0])
# 	# 	x, y, z = particle[1:4]
# 	# 	# new_item = np.array([x,y,z, p_size, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,0],dtype=np.int16).reshape(1,-1)
# 	# 	# new_item[0, 4 + p_class] = 1
# 	# 	new_item = [x,y,z, p_size, 1, p_class]
# 	# 	new_item = np.array(new_item)

# 	# 	label = np.vstack((label,new_item))

# 	# 	# print(label)
	
# 	# evaluate_shrec2021(pred_list=label,
# 	# 				base_dir=base_dir,data_id=0,class_num=13)
	
# 	# one_class = label[label[:,5] == 1]
# 	# print(one_class.shape)

# 	# with mrcfile.open('/storage_data/su_xiaofeng/shrec_data/shrec_2021/model_9/occupancy_mask.mrc',permissive=True) as f:
# 	# 	occupancy_map = f.data
	
# 	# gt_location = location
# 	# particle_dict = {'4V94_fixed': 1,
# 	# 				 '4CR2': 2,
# 	# 				 '1QVR': 3,
# 	# 				 '1BXN': 4,
# 	# 				 '3CF3': 5,
# 	# 				 '1U6G': 6,
# 	# 				 '3D2F': 7,
# 	# 				 '2CG9': 8,
# 	# 				 '3H84': 9,
# 	# 				 '3GL1': 10,
# 	# 				 '3QM1': 11,
# 	# 				 '1S3X': 12,
# 	# 				 '5MRC': 13,
# 	# 				 'fiducial': 14}

# 	# for i in range(len(gt_location)):
# 	# 	location.loc[i, 0] = particle_dict.get(location.loc[i, 0],0)

# 	# # result = occupancy_map[12,207,97].astype(np.int64)
	
# 	# particle_list = np.array(location.loc[:, 0:3])
# 	# gt_list = np.concatenate((np.array([[0, 0, 0, 0]]), particle_list), axis=0)
	
# 	# pred_x = one_class[:, 0]
# 	# pred_y = one_class[:, 1]
# 	# pred_z = one_class[:, 2]

# 	# result = occupancy_map[pred_z, pred_y, pred_x].astype(np.int64)

# 	### test evaluate 10045
# 	base_dir = '/storage_data/su_xiaofeng/relion/empiar-10045/build_targets/tomo_coords'
# 	location = pd.read_csv(os.path.join(base_dir, '11.txt'), header=None, sep='\t')
	
# 	label = np.zeros(shape=(0,6),dtype=np.int16)

# 	occupancy_map = mrcfile.open('/storage_data/su_xiaofeng/relion/empiar-10045/build_targets/tomo_occupancy/IS002_291013_011.mrc').data
	
# 	idx_list = []
# 	for j in range(100):
# 		x,y,z = location.loc[j]
# 		idx = occupancy_map[int(z),int(y),int(x)] 
# 		idx_list.append(idx)
	
# 	uni = np.unique(occupancy_map)
# 	print(uni)
# 	print(len(uni))
# 	print(np.min(uni))
# 	print(np.max(uni))


# 	print(idx_list)

# 	idx = np.unique(idx_list)
# 	print(np.min(idx_list))
# 	print(np.max(idx_list))
# 	print(len(idx))
# 	print(idx)


# 	print('----------')

		


