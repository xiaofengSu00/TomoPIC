import torch
import torch.nn as nn
from util.calculate import bbox_iou,bbox_eiou

def smooth_BCE(eps=0.1):
    """Returns label smoothing BCE targets for reducing overfitting; pos: `1.0 - 0.5*eps`, neg: `0.5*eps`. For details see https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441"""
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes FocalLoss with specified loss function, gamma, and alpha values; modifies loss reduction to
        'none'.
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Calculates the focal loss between predicted and true labels using a modified BCEWithLogitsLoss."""
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss_EIOU(nn.Module):
	def __init__(self, model,class_num, anchor_num, anchor,label_smoothing = 0):
		super().__init__()
		self.sort_obj_iou = False
		self.gr = 0
		device = next(model.parameters()).device

		## loss func
		cls_pw = 1
		obj_pw = 1
		BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cls_pw],device=device))
		BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([obj_pw],device=device))

        ## model para
		self.class_num = class_num
		self.anchor_num = anchor_num
		self.anchor = anchor
        
		# loss and loss weight
		self.cp,self.cn = smooth_BCE(eps = label_smoothing)
		self.BCEobj,self.BCEcls = BCEobj,BCEcls

		self.bbox_weight = 1#5#1
		self.obj_weight = 20 #100 #20 #100 #100
		self.cls_weight = 10

	def __call__(self, p, targets):
		ByteTensor = torch.cuda.ByteTensor if p.is_cuda else torch.ByteTensor
		FloatTensor = torch.cuda.FloatTensor if p.is_cuda else torch.FloatTensor

		loss_cls = FloatTensor(1).fill_(0)
		loss_bbox = FloatTensor(1).fill_(0)
		loss_obj = FloatTensor(1).fill_(0)
		
		indices,tbox,anchors,tcls = self.build_target(p, targets, scale_factor=4)
		
		bs_idx,an_idx,gk,gj,gi = indices
		
		tobj = torch.zeros(size =(p.shape[0],p.shape[1],p.shape[2],p.shape[3],p.shape[4])).type(FloatTensor)


		targte_num = bs_idx.shape[0]
		
		if targte_num:
			pred = p[bs_idx,an_idx,gk,gj,gi]

			# print(bs_idx)
			# print(bs_idx.unique())
			pred_xyz = pred[...,:3]
			pred_r = pred[...,3].view(-1,1)
			pred_cls = pred[...,5:]

			### compute bbox loss
			pred_xyz = pred_xyz.sigmoid()*2 - 0.5
			pred_r = (pred_r.sigmoid() * 2)**2 * anchors
			pred_box = torch.cat((pred_xyz,pred_r),1)

			iou = bbox_eiou(pred_box,tbox,EIOU=True,Focal=False)

			if type(iou) is tuple:
				loss_bbox += (iou[1].detach().squeeze() * (1 - iou[0].squeeze())).mean()
				iou = iou[0].squeeze()
			else:
				loss_bbox += (1.0 - iou.squeeze()).mean()  # iou loss
				iou = iou.squeeze()


			### object loss
			iou = iou.detach().clamp(0).type(tobj.dtype)

			if self.sort_obj_iou:
				j = iou.argsort()
				bs_idx, an_idx, gj, gi, iou = bs_idx[j], an_idx[j], gj[j], gi[j], iou[j]
			if self.gr < 1:
				iou = (1.0 - self.gr) + self.gr * iou
			# tobj[bs_idx, an_idx, gk,gj, gi] = iou  # iou ratio
			#tobj[bs_idx, an_idx, gk,gj, gi] = 1  # 如果发现预测的score不高 数据集目标太小太拥挤 困难样本过多 可以试试这个
			tobj[bs_idx, an_idx, gk,gj, gi] = iou
			### classification:
			# Classification
			if self.class_num > 1:  # cls loss (only if multiple classes)
				# t = torch.full_like(pred_cls, self.cn).type(FloatTensor)  # targets
				# t[range(targte_num), tcls] = self.cp
				# loss_cls +=  self.BCEcls(pred_cls, tcls)
				loss_cls += self.BCEcls(pred_cls,tcls) 

		loss_obj += self.BCEobj(p[...,4],tobj)

		loss_bbox *= self.bbox_weight
		loss_obj *= self.obj_weight
		loss_cls *= self.cls_weight
		batch_size = tobj.shape[0]

		return (loss_bbox+loss_obj+loss_cls)*batch_size,loss_obj,loss_cls

	def build_target(self, pred, targets, scale_factor):
		'''
		pred: x,y,z,r,conf,class
		output: targets ---- (x,y,z,r,conf,cls--12) + grid_x,grid_y,grid_z
		'''
		ByteTensor = torch.cuda.ByteTensor if pred.is_cuda else torch.ByteTensor
		FloatTensor = torch.cuda.FloatTensor if pred.is_cuda else torch.FloatTensor

		batch_size = pred.shape[0]
		max_target_num = targets.shape[1]

		### scale_factor, feature map size gz, gy, gx
		gain = ByteTensor(4).fill_(1)
		# gain = torch.ones(4, device=self.device)
		gain [1:] =  torch.tensor(pred.shape)[[2,3,4]] 
		gain[0] = scale_factor

		# if mask.any():

		### generate bs_index
		bs_idx = ByteTensor(targets.size(0), targets.size(1)).fill_(0)
		# bs_idx = torch.zeros((targets.size(0), targets.size(1)))
		for bs in range(targets.size(0)):
			bs_idx[bs, :] = bs
		## bs_targets----bs_idx,x,y,z,r,conf,class
		bs_targets = torch.cat((bs_idx.unsqueeze(-1),targets),dim=-1).view(batch_size*max_target_num, -1)
		mask = (targets[..., 4] > 0.5).view(batch_size*max_target_num)
		bs_targets = bs_targets[mask]

		# tcls, tbox, indices, anch = [], [], [], []

		tn = bs_targets.shape[0]

		# ai =(torch.arange(self.anchor_num))

		ai = torch.arange(self.anchor_num).float().view(self.anchor_num, 1).repeat(1, tn).type(FloatTensor)

		### repeat target according to anchor_num
		bs_targets = torch.cat((bs_targets.repeat(self.anchor_num, 1, 1), ai[..., None]), 2)  ## (3,3,18)
		# print(bs_targets)
		## 定义偏移量和偏置
		g = 0.5  # bias

		off = (
					torch.tensor(
						[
								[0, 0, 0],
								[1, 0, 0],
								[0, 1, 0],
								[0, 0, 1],
								[-1, 0, 0],
								[0, -1, 0],
								[0, 0, -1]  # j,k,l,m
							]
						).float()
						* g
			).type(FloatTensor)  # offsets
		# print(off)

		## 该层归一化后的 anchor  anchors (3,2)
		# self.anchor = torch.tensor([[5],[10],[15]])
		self.anchor = self.anchor.type(FloatTensor)
		anchor_norm = self.anchor[0, ...] / gain[0]

		### scale x,y,z acrroding to strides
		t = bs_targets.clone()
		t[..., 1:5] = t[..., 1:5] / gain[0]

		if tn:
			# inter = t[...,4]
			# print(inter.shape)
			# print(anchor_norm.shape)

			r = (t[..., 4] / anchor_norm).unsqueeze(-1)  ### (3,3,1) / (3,1,1) --- 3,3,1
			
			j = torch.max(r, 1 / r).max(2)[0] < 4  ## anchor_thre----4
			t = t[j]  ### (tn_after_filter,class_num +5 + 1)

			## offset
			gxyz = t[:, 1:4]
			gxyz_i = gain[1:] - gxyz
				

			a, b, c = ((gxyz % 1 < g) & (gxyz > 1)).T  ###
			d, e, f = ((gxyz_i % 1 < g) & (gxyz_i > 1)).T

			f = torch.stack((torch.ones_like(a), a, b, c, d, e, f))
			t = t.repeat((7, 1, 1))[f]

			offsets = (torch.zeros_like(gxyz)[None] + off[:, None])[f]
		

		else:
			t = bs_targets[0]
			offsets = 0

		b_idx,a_idx = t[...,0],t[...,-1]
		tcls = t[:, 6:-1]
		gxyz = t[:, 1:4]
		gr = t[:, 4]

		a_idx,b_idx =a_idx.long().view(-1), b_idx.long().view(-1)
		gijk = (gxyz - offsets).long()
		gi, gj, gk = gijk.T
		## pred_shape = (2,3,16,16,16,17)
		### bi ai xi,yi,zi
		indices = (b_idx, a_idx,gk.clamp_(0, gain[1] - 1), gj.clamp_(0, gain[2] - 1), gi.clamp_(0, gain[3] - 1))
		tbox = torch.cat((gxyz - gijk, gr.view(-1, 1)), 1)
		# anch = self.anchor[0,...][a_idx] ####  
		anch = anchor_norm[...][a_idx]

		return indices,tbox,anch,tcls
