import torch
import torch.optim as optim
import logging
import os
import sys

def get_logger(out_dir):
    logger = logging.getLogger('Exp')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_path = os.path.join(out_dir, "run.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger

## Optimizer
def initial_optim(decay_option, lr, weight_decay, net, optimizer, eps) : 
    
    if optimizer == 'adamw' : 
        optimizer_adam_family = optim.AdamW
    elif optimizer == 'adam' : 
        optimizer_adam_family = optim.Adam
    if decay_option == 'all':
        optimizer = optimizer_adam_family(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay, eps=eps)
        # optimizer = optimizer_adam_family(net.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=weight_decay, eps=eps)
        
    elif decay_option == 'noVQ':
        all_params = set(net.parameters())
        no_decay = set([net.vq_layer])
        
        decay = all_params - no_decay
        optimizer = optimizer_adam_family([
                    {'params': list(no_decay), 'weight_decay': 0}, 
                    {'params': list(decay), 'weight_decay' : weight_decay}], lr=lr, eps=eps)
        
    return optimizer
