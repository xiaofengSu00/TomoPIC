import argparse
import logging
import time
from tqdm import tqdm
from util.utils import *
from dataset.shrec2021 import SHREC2021
from model.yolo import Detect_Framework
from model.loss import ComputeLoss_EIOU
from eval.evaluate import evaluate_shrec2021
from util.calculate import *
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

def create_parser():
	parser = argparse.ArgumentParser()
	# project options
	parser.add_argument('--model_name', type=str, default='YOLO3D', help='Name of this experiment.')
	parser.add_argument('--gpu_ids', type=str, default='0', help='GPU ids, use -1 for CPU.')
	parser.add_argument('--save_dir', type=str, default='./checkpoints_2021', help='Models are saved here.')
	parser.add_argument('--load_dir', type=str, default='./checkpoints', help='The directory of the pretrained model.')
	"""important"""
	parser.add_argument('--dataset_dir', type=str,
						default='/storage_data/su_xiaofeng/shrec_data/shrec_2021',
						help='The directory of the used dataset')

	# training options
	parser.add_argument('--total_epoches', type=int, default=200, help='Total epoches.')
	parser.add_argument('--checkpoint_interval', type=int, default=10, help='Interval between saving model weights')
	parser.add_argument('--evaluation_interval', type=int, default=10,
						help='Interval between evaluations on validation set')
	parser.add_argument('--pretrained', type=bool, default=False, help='Use pretrained model.')
	parser.add_argument('--load_filename', type=str, default='YOLO3D_EPOCH[200].pth',
						help='Filename of the pretrained model.')

	# dataset options
	parser.add_argument("--batch_size", type=int, default=10, help="Size of each image batch.")
	parser.add_argument("--num_workers", type=int, default=4,
						help="number of cpu threads to use during batch generation")
	opt = parser.parse_args()

	return opt


def train(opt):
	from datetime import datetime

	device = prepare_devices(opt)

	
	train_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
	if not os.path.exists(os.path.join(opt.save_dir, train_time)):
		mkfolder(os.path.join(opt.save_dir, train_time))

	opt.save_dir = os.path.join(opt.save_dir, train_time)

	train_dataset = SHREC2021(mode='train', 
						   base_dir=opt.dataset_dir,aug_choice=4)

	train_data = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=opt.batch_size,
		shuffle=True,
		num_workers=opt.num_workers,
		collate_fn=train_dataset.collate_fn
	)

	eval_dataset = SHREC2021(mode='val', 
						  base_dir=opt.dataset_dir, aug_choice=0)
	eval_data = torch.utils.data.DataLoader(
		eval_dataset,
		batch_size=opt.batch_size,
		shuffle=False,
		num_workers=opt.num_workers,
		collate_fn=eval_dataset.collate_fn
	)

	"""
		Pre-cluster results.
		Need to be adjusted later.
	"""
	##########
	ANCHOR = torch.tensor([[[7], [13], [25]]])
	##########

	model = Detect_Framework(class_num=13, anchors=ANCHOR, anchor_num=3)

	device_ids = list(range(torch.cuda.device_count()))
	if opt.pretrained:
		model = load_model(model, opt)
	else:  # training from begining
		if opt.gpu_num > 1:
			model = nn.DataParallel(model, device_ids=device_ids).to(device)
		else:
			model = model.to(device)

	compute_loss = ComputeLoss_EIOU(model=model,
							   class_num=13,
							   anchor_num=3,
							   anchor=ANCHOR)
	
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
	 													   T_max=opt.total_epoches,
	 													   eta_min=1e-4)
	

	### early stop
	from util.training_tools import EarlyStopping

	early_stopping = EarlyStopping(save_path=opt.save_dir, patience=8)

	##########
	import pandas as pd

	his_df = pd.DataFrame([])
	val_df = pd.DataFrame([])
	train_loss_list = []
	train_loss_conf_list = []
	train_loss_cls_list = []

	val_loss_list = []
	val_loss_conf_list = []
	val_loss_cls_list = []

	start_time = time.time()
	for epoch in range(1, opt.total_epoches + 1):
		print('**********')
		print('Training Epoch %d' % epoch)
		print('Learning rate: %.4f' % scheduler.get_last_lr()[0])
		
		model.train()
		loss_list = []
		loss_conf_list = []
		loss_cls_list = []
		for batch_i, (imgs, targets) in enumerate(tqdm(train_data, desc=f"Epoch {epoch}")):
			imgs = imgs.to(device)
			targets = targets.to(device)

			train_out = model(imgs)
			loss, loss_conf, loss_cls = compute_loss(train_out, targets)
			loss = loss.mean()
			loss_conf = loss_conf.mean()
			loss_cls = loss_cls.mean()

			loss.backward()
			optimizer.step()
			optimizer.zero_grad()

			loss_list.append(loss.cpu().detach())
			loss_conf_list.append(loss_conf.cpu().detach())
			loss_cls_list.append(loss_cls.cpu().detach())

		# ------------
		# Log pregress
		# ------------
		train_loss_list.append(np.mean(loss_list))
		train_loss_conf_list.append(np.mean(loss_conf_list))
		train_loss_cls_list.append(np.mean(loss_cls_list))

		print('End of training epoch %d / %d \t Loss: %.6f \t Conf Loss: %.6f \t Class Loss: %.6f \n' % (
			epoch, opt.total_epoches, np.mean(loss_list), np.mean(loss_conf_list), np.mean(loss_cls_list)))

		# evaluation step
		if epoch % opt.evaluation_interval == 0:
			print("---- Evaluating Model ----")
			model.eval()

			##########
			conf_thres = 0.5
			nms_thres = 0.1

			##########

			loss_list = []
			loss_conf_list = []
			loss_cls_list = []

			with torch.no_grad():
				pred_list = []
				for batch_i, (imgs, targets) in enumerate(tqdm(eval_data, desc=f"Calculating")):
					imgs = imgs.to(device)
					targets = targets.to(device)
					train_out, pred = model(imgs)
					loss, loss_conf, loss_cls = compute_loss(train_out, targets)

					loss = loss.mean()
					loss_conf = loss_conf.mean()
					loss_cls = loss_cls.mean()
					loss_list.append(loss.cpu().detach())
					loss_conf_list.append(loss_conf.cpu().detach())
					loss_cls_list.append(loss_cls.cpu().detach())

					for b in range(pred.size(0)):
						mask = pred[b, ..., 4] > conf_thres

						b_pred = pred[b, mask, :]
						b_pred = b_pred.cpu()
						index = Non_Maximum_Suppression(b_pred[:, :5], nms_thres)
						b_pred = b_pred[index, :]
						pred_list.append(b_pred)

				result = eval_dataset.joint(pred_list)
				total_pred_list = result[0]
				final_pred_list = total_pred_list[remove(torch.from_numpy(total_pred_list[:, :5].astype(np.float32))),
								  :]

				print('EVAL Result:')
				print('Loss: %.6f \t Conf Loss: %.6f \t Class Loss: %.6f' % (
					np.mean(loss_list), np.mean(loss_conf_list), np.mean(loss_cls_list)))
				# print(final_pred_list[1:50,:])

				val_loss_list.append(np.mean(loss_list))
				val_loss_conf_list.append(np.mean(loss_conf_list))
				val_loss_cls_list.append(np.mean(loss_cls_list))

				fscore = evaluate_shrec2021(final_pred_list,data_id=8)
				early_stopping(fscore=fscore, model=model, opt=opt, epoch=epoch)

		if epoch % opt.checkpoint_interval == 0:
			save_model(model, opt, epoch)

		if early_stopping.early_stop:
			print(f'Early stopping at epoch {epoch}')
			break

		scheduler.step()

	end_time = time.time()
	time_duration = end_time - start_time
	from datetime import datetime

	his_df['total_loss'] = train_loss_list
	his_df['loss_conf'] = train_loss_conf_list
	his_df['loss_cls'] = train_loss_cls_list
	his_df.to_csv('./{}_training_loss.csv'.format(datetime.now().strftime("%Y-%m-%d-%H:%M:%S")))

	val_df['total_loss'] = val_loss_list
	val_df['loss_conf'] = val_loss_conf_list
	val_df['loss_cls'] = val_loss_cls_list
	val_df.to_csv('./{}_evaluate_loss.csv'.format(datetime.now().strftime("%Y-%m-%d-%H:%M:%S")))

	print('Train Time:{:.2f} hours'.format(time_duration/3600))

if __name__ == "__main__":
	opt = create_parser()
	train(opt)