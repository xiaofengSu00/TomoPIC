""" Functions for devices checking, model saving and loading.
"""

import os
import logging
import torch
import torch.nn as nn
import numpy as np

#from collections import OrderedDict
def normalize(x, percentile = True, pmin=4.0, pmax=96.0, axis=None, clip=False, eps=1e-20):
    """Percentile-based image normalization."""

    if percentile:
        mi = np.percentile(x,pmin,axis=axis,keepdims=True)
        ma = np.percentile(x,pmax,axis=axis,keepdims=True)
        out = (x - mi) / ( ma - mi + eps )
        out = out.astype(np.float32)
        if clip:
            return np.clip(out,0,1)
        else:
            return out
    else:
        out = (x-np.mean(x))/np.std(x)
        out = out.astype(np.float32)
        return out

def mkfolder(folder):
    import os
    try:
        os.makedirs(folder)
    except FileExistsError:
        logging.warning("The {0} folder already exists  \n The old {0} folder will be renamed (to {0}~)".format(folder))
        import shutil
        if os.path.exists(folder+'~'):
            shutil.rmtree(folder+'~')
        os.rename('{}'.format(folder), '{}'.format( folder+'~'))
        os.makedirs(folder)

def prepare_devices(opt):
    """ Prepare device for training or testing.
        Using cpu is not recommended.
    """
    if opt.gpu_ids != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
        if torch.cuda.is_available():
            opt.gpu_num = torch.cuda.device_count()#used by function below
            device = torch.device("cuda:0")
            print('Using %s GPU(s) for this project.' % torch.cuda.device_count())
        else:
            device = torch.device("cpu")
            opt.gpu_num = 0
            print('CUDA is unavailable, using CPU only.')
    else:
        device = torch.device("cpu")
        opt.gpu_num = 0
        print('Using CPU only.')
    
    opt.device = device
    return device


def save_model(model, opt, epoch,is_best=False):
    """ Save the patameters of the model.
        Always save model without "module" (just on one device).
    """
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    
    if is_best:
        save_filename = '%s_EPOCH[%s]_best.pth' % (opt.model_name, epoch)
    else:
        save_filename = '%s_EPOCH[%s].pth' % (opt.model_name, epoch)
    save_path = os.path.join(opt.save_dir, save_filename)
    if opt.gpu_num > 1:
        torch.save(model.module.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)



def load_model(model, opt):
    """ Load the parameters of the model.
        DO EVERYTHING! No need to care about the model.
    """
    load_path = os.path.join(opt.load_dir, opt.load_filename)
    model.load_state_dict(torch.load(load_path, map_location=opt.device))

    if opt.gpu_num > 1:
        model = nn.DataParallel(model)
    model = model.to(opt.device)
    
    return model