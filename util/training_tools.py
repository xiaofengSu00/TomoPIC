import os
import torch 
import numpy as np
import torch
import math
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler


import os
import numpy as np
import torch

class ModelCheckpoint:
    """Saves the best model based on training loss."""
    
    def __init__(self, save_path, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            verbose (bool): If True, prints a message for each model saving event. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.verbose = verbose
        self.best_train_loss = np.Inf  # Initialize best training loss as infinity
        self.delta = delta

    def __call__(self, train_loss, model, opt, epoch):
        """
        Save the model if the training loss improves.
        
        Args:
            train_loss (float): Current training loss.
            model: The model to be saved.
            opt: Options or configurations used for saving the model.
            epoch (int): Current epoch number.
        """
        # Check if the current training loss is lower than the best training loss
        if train_loss < self.best_train_loss - self.delta:
            self.best_train_loss = train_loss
            self.save_checkpoint(train_loss, model, opt, epoch)

    def save_checkpoint(self, train_loss, model, opt, epoch):
        """Saves the model when training loss decreases."""
        if self.verbose:
            print(f'Training loss improved ({self.best_train_loss:.6f} --> {train_loss:.6f}). Saving model ...')
        
        path = os.path.join(self.save_path, 'best_network.pth')
        
        # Save the model's state_dict
        if opt.gpu_num > 1:
            torch.save(model.module.state_dict(), path)
        else:
            torch.save(model.state_dict(), path)

# Example usage:
# checkpoint = ModelCheckpoint(save_path='your_save_path', verbose=True)
# for epoch in range(num_epochs):
#     train_loss = train(...)  # Your training logic here
#     checkpoint(train_loss, model, opt, epoch)



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=3, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_fscore_max = np.Inf
        self.delta = delta

    def __call__(self, fscore, model,opt,epoch):

        score = fscore  ### wgy negtive

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(fscore, model,opt,epoch)
            
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(fscore, model,opt,epoch)
            self.counter = 0

    def save_checkpoint(self, val_fscore, model,opt,epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation f1 decreased ({self.val_fscore_max:.6f} --> {val_fscore:.6f}).  Saving model ...')
        # path = os.path.join(self.save_path, 'epoch[{}]_f1[{:.4f}]_best_network.pth'.format(epoch,val_f1))
        path = os.path.join(self.save_path, 'best_network.pth')
        if opt.gpu_num>1:
            torch.save(model.module.state_dict(), path)
        else:
            torch.save(model.state_dict(), path)	# save
        # save_model(model,opt,epoch=epoch)
        self.val_fscore_max = val_fscore



### warm up
class WarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, steps_per_epoch, min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.max_steps = max_epochs * steps_per_epoch
        self.min_lr = min_lr
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            # Linear warmup
            lr_scale = step / self.warmup_steps
        else:
            # Cosine annealing
            progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            lr_scale = 0.5 * (1 + torch.cos(torch.tensor(progress * math.pi)))
        return [base_lr * lr_scale for base_lr in self.base_lrs]


if __name__ == '__main__':
    import mrcfile
    with mrcfile.open('/storage_data/su_xiaofeng/relion/empiar-10045/Tomograms/tomo005/IS002_291013_005.mrc',permissive=True) as f:
        vx = f.voxel_size
        print(vx)
        
    # 早停止
    # early_stopping = EarlyStopping(save_path='')
    # early_stopping(eval_loss, model)
    # #达到早停止条件时，early_stop会被置为True
    # if early_stopping.early_stop:
    #     print("Early stopping")
    #     break #跳出迭代，结束训练