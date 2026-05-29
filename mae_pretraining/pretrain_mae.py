
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import math
from PIL import Image
from skimage import io, color, segmentation

import glob, os
from time import time
from natsort import natsorted
import logging
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp

from Model.utils import adjust_learning_rate_halfcosine, save_model


from torch.utils.data import DataLoader


def pretrain_mae(cfg = cfg, model = model, optimizer = optimizer, scaler = scaler, logger = logger, 
                checkpoint_path=checkpoint_path,validation_dataloader = validation_dataloader, pretraining_dataloader = pretraining_dataloader,
                pretraining_dataset = pretraining_dataset):

    "Do Mae pretraining"

    # Read log period
    log_period = cfg['TRAINING']['LOGGING_PERIOD']
    batch_size = cfg['DATALOADER']['BATCH_SIZE']
    iter_per_epoch = len(pretraining_dataset) / batch_size
    epochs = cfg['TRAINING']['EPOCHS']
    mask_ratio = cfg['TRAINING']['MASK_RATIO']
    
    logger.info('Started training')

    # Train the Model
    batch_time, net_time = [], []

    iter_start = args.iter_start
    steps = args.iter_start
    best_val_loss = float('inf')
    best_epoch    = -1
    
    # performance metrics helpers
    average_loss = 0

    for epoch in range(int(iter_start/iter_per_epoch), epochs):
        model.train()
        average_train_loss = 0.
        end = time()

        for batch_data in pretraining_dataloader:
            batch_time.append(time()-end)
            if len(batch_time)>100:
                del batch_time[0]

            adjust_learning_rate_halfcosine(optimizer, steps / len(pretraining_dataloader) + epoch, cfg)

            optimizer.zero_grad(set_to_none=True)
            t = time()
        
            images = batch_data["image"].cuda(non_blocking=True)   
            # with amp.autocast(enabled=True):
            loss, _, _ = model(images, mask_ratio=args.mask_ratio)
            # print('loss:', loss)
        
                    
            # Forward + Backward + Optimize
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

            lr = optimizer.param_groups[0]["lr"]

            torch.cuda.synchronize()

            net_time.append(time()-t)
            if len(net_time)>100:
                del net_time[0]

            # other way to calculate accuracy
            average_loss += float(loss.item())
            # writer.add_scalar(SAVE_DIR + '_loss', float(loss.item()), steps)

            steps += 1
            wandb.log({"pretrain train loss": loss.item()})
            
            end = time()
        
        # ------------------------------------------------------------------ #
        #  Validation                                                          #
        # ------------------------------------------------------------------ #
        model.eval()
        average_val_loss = 0.
        n_val_batches    = 0

        with torch.no_grad():
            for batch_data in pretraining_val_dataloader:
                images = batch_data["image"].cuda(non_blocking=True)
                #with amp.autocast(enabled=True):
                loss, _, _ = model(images, mask_ratio=mask_ratio)
                average_val_loss += float(loss.item())
                n_val_batches    += 1

        average_val_loss /= n_val_batches

        # ------------------------------------------------------------------ #
        #  Logging                                                             #
        # ------------------------------------------------------------------ #
        logger.info(
            '[%2d/%2d] %5d) [batch load %2.3fs, net %1.2fs], '
            'LR %.6f, Train Loss: %1.3f, Val Loss: %1.3f'
            % (epoch + 1, epochs, steps,
               np.mean(batch_time), np.mean(net_time),
               lr, average_train_loss, average_val_loss)
        )

        wandb.log({
            "pretrain/train_loss_epoch": average_train_loss,
            "pretrain/val_loss_epoch":   average_val_loss,
            "pretrain/lr":               lr,
            "epoch":                     epoch,
        })

        # ------------------------------------------------------------------ #
        #  Checkpointing                                                       #
        # ------------------------------------------------------------------ #
        # always save latest
        # save_model(args, cfg, model,
        #            os.path.join(SAVE_DIR, 'checkpoint_latest'),
        #            epoch, steps)

        # save best based on val loss
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            best_epoch    = epoch
            save_model(args, cfg, model,checkpoint_path,
                       epoch, steps)
            logger.info(
                f'New best val loss: {best_val_loss:.4f} at epoch {epoch + 1}'
            )

        logger.info(
            f'Best so far: epoch {best_epoch + 1}, val loss {best_val_loss:.4f}'
        )

        # early stopping hooks 
        if os.path.exists(os.path.join(cfg['TRAINING']['CHECKPOINT'], 'stop.txt')):
            logger.info('Stop file detected - ending training early.')
            break

        if os.path.exists(os.path.join(cfg['TRAINING']['CHECKPOINT'], 'pdb.txt')):
            import pdb; pdb.set_trace()

    logger.info(
        f'Training finished. Best checkpoint: epoch {best_epoch + 1}, '
        f'val loss {best_val_loss:.4f}'
    )
        
        