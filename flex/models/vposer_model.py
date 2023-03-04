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
#
# If you use this code in a research publication please consider citing the following:
#
# Expressive Body Capture: 3D Hands, Face, and Body from a Single Image <https://arxiv.org/abs/1904.05866>
#
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2020.12.12

import numpy as np
import torch
from flex.tools.angle_continuous_repres import geodesic_loss_R
from flex.models.body_model import BodyModel
from flex.tools.rotation_tools import aa2matrot
from flex.models.model_components import BatchFlatten
from flex.tools.rotation_tools import matrot2aa
from flex.tools.registry import registry
from torch import nn
from torch.nn import functional as F
from functools import partial
import math


# For debugging only.
def hook(name, grad):
    index_tuple = (grad==math.inf).nonzero(as_tuple=True)
    if len(index_tuple[0]) > 0:
        print(name, grad.shape, grad.max(), grad.min(), grad.sum())
    grad[index_tuple] = 0

    index_tuple = (grad==-math.inf).nonzero(as_tuple=True)
    if len(index_tuple[0]) > 0:
        print(name, grad.shape, grad.max(), grad.min(), grad.sum())
    grad[index_tuple] = 0
    return grad


class ContinousRotReprDecoder(nn.Module):
    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)


class NormalDistDecoder(nn.Module):
    def __init__(self, num_feat_in, latentD):
        super(NormalDistDecoder, self).__init__()

        self.mu = nn.Linear(num_feat_in, latentD)
        self.logvar = nn.Linear(num_feat_in, latentD)

    def forward(self, Xout):
        return torch.distributions.normal.Normal(self.mu(Xout), F.softplus(self.logvar(Xout)))


@registry.register_class(name='VPoser')
class VPoser(nn.Module):
    def __init__(self, model_ps):
        super(VPoser, self).__init__()

        self.vp_ps = model_ps
        self.bm_train = BodyModel(f'{model_ps.smplx_dir}/smplx/SMPLX_NEUTRAL.npz')

        num_neurons, self.latentD = model_ps.num_neurons, model_ps.latentD

        self.num_joints = 21
        n_features = self.num_joints * 3

        self.encoder_net = nn.Sequential(
            BatchFlatten(),
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, num_neurons),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_neurons),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons),
            nn.Linear(num_neurons, num_neurons),
            NormalDistDecoder(num_neurons, self.latentD)
        )

        self.decoder_net = nn.Sequential(
            nn.Linear(self.latentD, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons),
            nn.LeakyReLU(),
            nn.Linear(num_neurons, self.num_joints * 6),
            ContinousRotReprDecoder(),
        )

    def encode(self, pose_body):
        '''
        :param Pin: Nx(numjoints*3)
        :param rep_type: 'matrot'/'aa' for matrix rotations or axis-angle
        :return:
        '''
        return self.encoder_net(pose_body)

    def decode(self, Zin):
        bs = Zin.shape[0]

        prec = self.decoder_net(Zin)
        if prec.requires_grad:
            prec.register_hook(partial(hook, 'prec'))

        prec1 = matrot2aa(prec).reshape(bs, -1, 3)
        if prec1.requires_grad:
            prec1.register_hook(partial(hook, 'prec1'))

        return {
            'pose_body': prec1.reshape(bs, -1, 3),
            'pose_body_matrot': prec.reshape(bs, -1, 9)
        }


    def forward(self, pose_body):
        '''
        :param Pin: aa: Nx1xnum_jointsx3 / matrot: Nx1xnum_jointsx9
        :param input_type: matrot / aa for matrix rotations or axis angles
        :param output_type: matrot / aa
        :return:
        '''

        q_z = self.encode(pose_body)
        q_z_sample = q_z.rsample()
        decode_results = self.decode(q_z_sample)
        decode_results.update({'poZ_body_mean': q_z.mean, 'poZ_body_std': q_z.scale, 'q_z': q_z})
        return decode_results

    def sample_poses(self, num_poses):
        some_weight = [a for a in self.parameters()][0]
        dtype = some_weight.dtype
        device = some_weight.device
        self.eval()
        with torch.no_grad():
            Zgen = torch.tensor(np.random.normal(0., 1., size=(num_poses, self.latentD)), dtype=dtype, device=device)

        out = self.decode(Zgen)
        out['latent'] = Zgen
        return out

    def loss_function(self, dorig, drec, current_epoch, mode):
        l1_loss = torch.nn.L1Loss(reduction='mean')
        geodesic_loss = geodesic_loss_R(reduction='mean')

        bs, latentD = drec['poZ_body_mean'].shape
        device = drec['poZ_body_mean'].device

        loss_kl_wt = self.vp_ps.loss_kl_wt
        loss_rec_wt = self.vp_ps.loss_rec_wt
        loss_matrot_wt = self.vp_ps.loss_matrot_wt
        loss_jtr_wt = self.vp_ps.loss_jtr_wt

        # q_z = torch.distributions.normal.Normal(drec['mean'], drec['std'])
        q_z = drec['q_z']
        # dorig['fullpose'] = torch.cat([dorig['root_orient'], dorig['pose_body']], dim=-1)

        # Reconstruction loss - L1 on the output mesh
        # TODO: Check that this BodyModel is the same as rendering through neutral smplx model.
        self.bm_train.to(device)
        with torch.no_grad():
            bm_orig = self.bm_train(pose_body=dorig)
        bm_rec = self.bm_train(pose_body=drec['pose_body'].contiguous().view(bs, -1))

        v2v = l1_loss(bm_rec.v, bm_orig.v)

        # KL loss
        p_z = torch.distributions.normal.Normal(
            loc=torch.zeros((bs, latentD), device=device, requires_grad=False),
            scale=torch.ones((bs, latentD), device=device, requires_grad=False))
        weighted_loss_dict = {
            'loss_kl':loss_kl_wt * torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_z, p_z), dim=[1])),
            'loss_mesh_rec': loss_rec_wt * v2v
        }

        if (current_epoch < self.vp_ps.keep_extra_loss_terms_until_epoch):
            weighted_loss_dict['matrot'] = loss_matrot_wt * geodesic_loss(drec['pose_body_matrot'].view(-1,3,3), aa2matrot(dorig.view(-1, 3)))
            weighted_loss_dict['jtr'] = loss_jtr_wt * l1_loss(bm_rec.Jtr, bm_orig.Jtr)

        weighted_loss_dict['loss_total'] = torch.stack(list(weighted_loss_dict.values())).sum()

        with torch.no_grad():
            unweighted_loss_dict = {'v2v': torch.sqrt(torch.pow(bm_rec.v - bm_orig.v, 2).sum(-1)).mean()}
            unweighted_loss_dict['loss_total'] = torch.cat(
                list({k: v.view(-1) for k, v in unweighted_loss_dict.items()}.values()), dim=-1).sum().view(1)

        if mode == 'train':
            if (current_epoch < self.vp_ps.keep_extra_loss_terms_until_epoch):
                return weighted_loss_dict['loss_total'], {'loss/total': weighted_loss_dict['loss_total'],
                                                          'loss/KLD': weighted_loss_dict['loss_kl'],
                                                          'loss/Mesh_rec': weighted_loss_dict['loss_mesh_rec'],
                                                          'loss/MatRot': weighted_loss_dict['matrot'],
                                                          'loss/Jtr': weighted_loss_dict['jtr']}
            return weighted_loss_dict['loss_total'], {'loss/total': weighted_loss_dict['loss_total'],
                                                      'loss/KLD': weighted_loss_dict['loss_kl'],
                                                      'loss/Mesh_rec': weighted_loss_dict['loss_mesh_rec']}
        return unweighted_loss_dict['loss_total'], {'loss/total': unweighted_loss_dict['loss_total'],
                                                    'loss/Mesh_rec': unweighted_loss_dict['v2v']}