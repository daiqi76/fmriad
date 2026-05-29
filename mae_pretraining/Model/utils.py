import os
import glob
import numpy as np

import torch
from sklearn.metrics import roc_auc_score
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler

import timm


class EarlyStopping:

def load_pretrained_checkpoint(model, checkpoint_path):
    files = [f for f in os.listdir(checkpoint_path) if filename in f]
    if len(files)>0:
        files.sort()
        ckp = files[-1]
        model.load_state_dict(torch.load(checkpoint_path+ckp)['net'])
        print(ckp, ' found and loaded.')
    
    return model

def make_scheduler():
    pass

def adjust_alpha():
    pass

def set_requires_grad():
    pass

def loop_iterable(iterable):
    pass

def save_model(args, cfg, model, filename, epoch, steps):
    flist = glob.glob(filename+ '*')
    for f in flist:
        os.remove(f)
    filename = filename + '_%03i_%06d.pth.tar'%(epoch, steps)
    if len([x for x in args.devices.split(",")]) > 1:
        state = {"net": model.module.state_dict()}
    else:
        state = {"net": model.state_dict()}
    torch.save(state, filename)

def adjust_learning_rate_halfcosine(optimizer, epoch, cfg):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < cfg['SOLVER']['warmup_epochs']:
        lr = cfg['SOLVER']['lr'] * epoch / cfg['SOLVER']['warmup_epochs'] 
    else:
        lr = cfg['SOLVER']['min_lr'] + (cfg['SOLVER']['lr'] - cfg['SOLVER']['min_lr']) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - cfg['SOLVER']['warmup_epochs']) / (cfg['TRAINING']['EPOCHS'] - cfg['SOLVER']['warmup_epochs'])))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, dist_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if dist_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), np.zeros([1, embed_dim]), pos_embed], axis=0)
    elif cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    # use half of dimensions to encode each axis
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb