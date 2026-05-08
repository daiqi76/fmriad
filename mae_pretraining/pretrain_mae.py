


def pretrain_mae(cfg = cfg, model = model, optimizer = optimizer, scaler = scaler, logger = logger, 
                 SAVE_DIR = SAVE_DIR,pretraining_dataloader = pretraining_dataloader,
                pretraining_dataset = pretraining_dataset):
    