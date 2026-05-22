from maskedautoencoder import MaskedAutoencoderViT
import torch.optim as optim

def make_optimizer(cfg,args, model):
    """
    Initialize an optimizer based on the configs of cfg and args

    Parameters:
    -----------
    cfg, args: config and argument parser from the command line
    model: torch.nn

    Returns:
    ---------
    optimizer: torch.optim.AdamW

    """
    optimizer = torch.optim.AdamW(
                                model.parameters(),
                                lr=cfg['SOLVER']['lr'],
                                weight_decay=cfg['SOLVER']['weight_decay'],
                                betas=(cfg['SOLVER']['beta1'], cfg['SOLVER']['beta2'])
                                )
    
    return optimizer
    
def build_mae(cfg, args):
    model_mae = MaskedAutoencoderViT(
        img_size          = cfg['MODEL']['img_size'],
        patch_size        = cfg['MODEL']['patch_size'], 
        in_chans          = cfg['MODEL']['in_chans'],
        embed_dim         = cfg['MODEL']['embed_dim'], 
        depth             = cfg['MODEL']['depth'], 
        num_heads         = cfg['MODEL']['n_heads'],
        qkv_bias          = cfg['MODEL']['qkv_bias'],
        drop_path_rate    = cfg['MODEL']['drop_path_rate'],
        decoder_embed_dim = cfg['MODEL']['decoder_embed_dim'], 
        decoder_depth     = cfg['MODEL']['decoder_depth'], 
        decoder_num_heads = cfg['MODEL']['decoder_num_heads'],
        mlp_ratio         = cfg['MODEL']['mlp_ratio'], 
        norm_pix_loss     = cfg['MODEL']['norm_pix_loss']
    )

    print('MAE model built.')
    return model_mae