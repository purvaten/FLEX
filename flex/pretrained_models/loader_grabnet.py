# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
import torch
import mano

from flex.models.grabnet_model import CoarseNet, RefineNet


class Trainer:

    def __init__(self, cfg):

        # Setup cuda.
        torch.manual_seed(cfg.seed)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.empty_cache()
        self.device = torch.device("cuda:%d" % cfg.cuda_id if cfg.cuda_id != -1 else "cpu")

        with torch.no_grad():
            rhm_train = mano.load(model_path=cfg.rhm_path,
                                       model_type='mano',
                                       num_pca_comps=45,
                                       batch_size=cfg.batch_size,
                                       flat_hand_mean=True).to(self.device)
        
        # Define model.
        self.coarse_net = CoarseNet().to(self.device).eval()
        self.refine_net = RefineNet().to(self.device).eval()
        self.refine_net.rhm_train = rhm_train

        # Display trainable parameter count.
        vars_cnet = [var[1] for var in self.coarse_net.named_parameters()]
        vars_rnet = [var[1] for var in self.refine_net.named_parameters()]
        cnet_n_params = sum(p.numel() for p in vars_cnet if p.requires_grad)
        rnet_n_params = sum(p.numel() for p in vars_rnet if p.requires_grad)
        print('\nTotal Trainable Parameters for Generative Hand Grasping Model (CoarseNet) is %2.2f M.' % ((cnet_n_params) * 1e-6))
        print('Total Trainable Parameters for Generative Hand Grasping Model (RefineNet) is %2.2f M.' % ((rnet_n_params) * 1e-6))

        # Initialize with pre-trained model.
        if cfg.best_cnet is not None:
            self.coarse_net.load_state_dict(torch.load(cfg.best_cnet, map_location=self.device), strict=False)
            print('------> Loading CoarseNet model (pre-trained on GRAB right-hand grasps) from %s' % cfg.best_cnet)
        if cfg.best_rnet is not None:
            pretrained_dict = torch.load(cfg.best_rnet, map_location=self.device)
            x = {k:v for k, v in pretrained_dict.items() if k not in ['rhm_train.betas', 'rhm_train.global_orient', 'rhm_train.transl', 'rhm_train.hand_pose']}
            self.refine_net.load_state_dict(x, strict=False)
            print('------> Loading RefineNet model (pre-trained on GRAB right-hand grasps) from %s\n' % cfg.best_rnet)
