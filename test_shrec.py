import argparse
import torch
import numpy as np
from tqdm import tqdm
import time
from util.utils import *
from eval.evaluate import *
from dataset.shrec2020 import SHREC2020
from dataset.shrec2021 import SHREC2021
from model.yolo import Detect_Framework
from model.loss import ComputeLoss_EIOU
from util.calculate import *
import time
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

start = time.time()

def test_shrec2021(opt):
    
    ##########preparing
    device = prepare_devices(opt)

    eval_dataset = SHREC2021(mode='test', base_dir=opt.dataset_dir,
                             aug_choice=0)
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
    ##########
    ANCHOR = torch.tensor([[[7], [13], [25]]])
   
    ##########
    
    model = Detect_Framework( class_num=13, anchors=ANCHOR,anchor_num=3)
    model = load_model(model, opt)


    compute_loss = ComputeLoss_EIOU(model = model,
                 class_num=13, 
                 anchor_num=3, 
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

        fscore = evaluate_shrec2021(final_pred_list,base_dir=opt.dataset_dir,data_id=9)

        df = pd.DataFrame(final_pred_list)
        df.to_csv('./shrec2021_result.csv',sep=',',header=False,index=False)

        end = time.time()
        time_duration = end-start

        print('Prediction Time:{:.2f} s'.format(time_duration))


def test_shrec2020(opt):
    
    ##########preparing
    device = prepare_devices(opt)

    eval_dataset = SHREC2020(mode='test', base_dir=opt.dataset_dir,
                             aug_choice=0)
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
    ##########

    ANCHOR = torch.tensor([[[5], [10], [15]]])

    ##########
    
    model = Detect_Framework( class_num=12, anchors=ANCHOR,anchor_num=3)
    model = load_model(model, opt)
  

    compute_loss = ComputeLoss_EIOU(model = model,
                class_num=12, 
                anchor_num=3, 
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
        print(final_pred_list)
        end_time = time.time()
        epoch_duration = end_time - start_time
        print('Test Time:{:.2f}'.format(epoch_duration))
        print('EVAL Result:')
        print('Loss: %.6f \t Conf Loss: %.6f \t Class Loss: %.6f' % (np.mean(loss_list), np.mean(loss_conf_list), np.mean(loss_cls_list)))
        
        fscore = evaluate_shrec2020(final_pred_list,base_dir=opt.dataset_dir,data_id=9)

        df = pd.DataFrame(final_pred_list)
        df.to_csv('./shrec2020_result.csv',sep=',',header=False,index=False)

        end = time.time()
        time_duration = end-start

        print('Prediction Time:{:.2f} s'.format(time_duration))


def create_parser():
    parser = argparse.ArgumentParser()

    #project options
    parser.add_argument('--model_name', type=str, default='YOLOv5', help='Name of this experiment.')
    parser.add_argument('--gpu_ids', type=str, default='0', help='GPU ids, use -1 for CPU.')

    parser.add_argument('--load_dir', type=str, default='./checkpoints_2021', help='The directory of the pretrained model.')
    
    parser.add_argument('--dataset_dir', type=str, default='/storage_data/su_xiaofeng/shrec_data/shrec_2021', help='The directory of the used dataset')
    
    #testing options
    parser.add_argument('--load_filename', type=str, default='best_network.pth', help='Filename of the pretrained model.')
   
    #dataset options
    parser.add_argument("--batch_size", type=int, default=10, help="Size of each image batch.")
    parser.add_argument("--num_workers", type=int, default=4, help="number of cpu threads to use during batch generation")
    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    opt = create_parser()
    test_shrec2020(opt)
    # test_shrec2021(opt)

