"""
Main class for data loading trained model and data.
"""
import torch

from flex.tools.vposer_model_loader import load_model
from flex.models.vposer_model import VPoser
from flex.tools.registry import registry
from pathlib import Path


class Trainer:

    def __init__(self, cfg):        
        # Setup cuda.
        self.device = "cuda:%d" % cfg.cuda_id if cfg.cuda_id != -1 else "cpu"

        # Define model.
        self.mime_net = registry.get_class('VPoser')(cfg).to(self.device)

        # Display trainable parameter count.
        vars_mnet = [var[1] for var in self.mime_net.named_parameters()]
        mnet_n_params = sum(p.numel() for p in vars_mnet if p.requires_grad)
        print('\nTotal Trainable Parameters for Human Generative Model (VPoser) is %2.2f M.' % ((mnet_n_params) * 1e-6))

        # Initialize with pre-trained model.
        if cfg.vposer_dset == 'amass':
            expr_dir = str(Path(cfg.best_mnet_amass).parents[1])
            self.mime_net, _ = load_model(expr_dir, model_code=VPoser, custom_ps=cfg,
                                        remove_words_in_model_weights='vp_model.',
                                        disable_grad=False, device=self.device)
            self.mime_net.to(self.device)
            print('------> Loading VPoser model (pre-trained on AMASS) from %s\n' % cfg.best_mnet_amass)
        else:  # grab
            state = torch.load(cfg.best_mnet_grab, map_location=self.device)
            self.mime_net.load_state_dict(state['state_dict'], strict=False)
            print('------> Restored VPoser model (pre-trained on GRAB) from %s\n' % cfg.best_mnet_grab)
