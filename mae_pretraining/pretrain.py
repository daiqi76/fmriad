
from pretrain_mae import pretrain_mae
from masked_autoencoder import build_mae,make_optimizer
from dataloader import build_pretraining_dataloader
from datetime import datetime
import argparse
import os
import random
import numpy as np
import torch
import yaml
import logging
import wandb

def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")
    
    
def wandb_setup(cfg, args, SAVE_DIR):
    # start a new wandb run to track this script
    os.makedirs(SAVE_DIR+'/'+'./wandb/', exist_ok=True)
    wandb.init(
        # set the wandb project where this run will be logged
        project="MAE Pre-training",
        name=SAVE_DIR,
        
        # track hyperparameters and run metadata
        config={
        "config_file": args.config_file,
        "lr": cfg['SOLVER']['lr'],
        "dir": SAVE_DIR+'/'+'./wandb/',
        "mask ratio": args.mask_ratio,
        "seed": args.seed
        }
    )

def setup_logger(SAVE_DIR, timestamp_current):
    # Set up the logger
    if len(glob.glob( SAVE_DIR +'/logs/'+ '*')) > 0:
        print('Logger found...')
        print('----------------')
        logging.basicConfig(filename=glob.glob(SAVE_DIR + '/logs/' + '*')[-1],
                            format='%(asctime)s %(message)s',
                            filemode='a',
                            level=logging.DEBUG, 
                            force=True)
    else:
        logging.basicConfig(filename=SAVE_DIR + '/logs/' + SAVE_DIR + '_' + timestamp_current + '.log', 
                        format='%(asctime)s %(message)s',
                        filemode='w',
                        level=logging.DEBUG, 
                        force=True)
    logger = logging.getLogger()
    return logger

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Pretrain a Masked Autoencoder (MAE) on fMRI data.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the directory containing the fMRI data.")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to the directory where the pretrained model will be saved.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration file for training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--mask_ratio", type=float, default=0.75, help="Ratio of the input to mask during pretraining.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
    
    args = parser.parse_args()
    
    config_file = open(args.config_file, 'rb')
    cfg = yaml.load(config_file, Loader=yaml.FullLoader)
    
    set_seed(args.seed)
    
    base_directory = "/home/hpc/iwi5/iwi5360h/FMRIAD/mae_pretraining/"
    SAVE_DIR = base_directory + args.save_dir + '_' +  '_seed_' + str(args.seed)

    # Logging
    
    timestamp_current = datetime.now()
    timestamp_current = timestamp_current.strftime("%Y%m%d_%H%M")

    # Set up the Tensorboard log
    # writer = setup_tensorboard(FILENAME_POSTFIX, timestamp_current)
    # Set up logger file
    logger = setup_logger(SAVE_DIR, timestamp_current)
    logger.setLevel(logging.DEBUG)
    logger.info('Process number: %d'%(os.getpid()))
    logger.info("Started training. Savename : " + args.save_dir)
    logger.info("Seed : " + str(args.seed))

    # Init wandb
    wandb_setup(cfg, args, SAVE_DIR)
    
    # Build the MAE model
    model = build_mae(cfg, args)
    print("Model built successfully.")
    model.cuda()
    
    # Build Dataset and Dataloader
    pretraining_dataset, pretraining_dataloader = build_pretraining_dataloader(cfg, args)
    
    # Build the optimizer
    
    optimizer = make_optimizer(cfg,args, model)
    print("Optimizer created successfully.")
    scaler = torch.cuda.amp.GradScaler()
    
    logger.info(cfg)
    logger.info(args)
    
    pretrain_mae(cfg = cfg,
                model = model,
                optimizer = optimizer,
                scaler = scaler,
                logger = logger,
                SAVE_DIR = SAVE_DIR,
                pretraining_dataloader = pretraining_dataloader,
                pretraining_dataset = pretraining_dataset)