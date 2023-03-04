r"""
Main file: Given an object, perform search for body-translation, body-orientation, latent (z) of VPoser & latent (w) of GrabNet that can minimize loss.
This is set up for obstacles in GRAB frame-of-reference (converted from Habitat environment).

NOTE:
- Results are saved in GRAB frame-of-reference.
- See visualization of data and results together in `notebooks/viz_results.ipynb`

CUDA_VISIBLE_DEVICES=0 python run.py \
--obj_name stapler \
--receptacle_name receptacle_aabb_TvStnd1_Top3_frl_apartment_tvstand \
--ornt_name all \
--index 0 \
--gender 'female'
"""
from flex.tools.utils import load_obj_verts, replace_topk, aa2rotmat, rotmat2aa, recompose_angle, euler_torch
from flex.pretrained_models.loader_grabnet import Trainer as RHGrasp_Loader
from flex.pretrained_models.loader_vposer import Trainer as VPoser_Loader
from flex.tto.inf_opt import optimize_findz
from flex.tools.config import Config

from bps_torch.bps import bps_torch
from omegaconf import OmegaConf
from datetime import datetime
from psbody.mesh import Mesh
import numpy as np
import argparse
import torch
import time
import math
import os

import random
random.seed(12345)


# =============Main classes====================================================================
class Optimize(RHGrasp_Loader, VPoser_Loader):

    def __init__(self, cfg_vp, cfg_rh):
        RHGrasp_Loader.__init__(self, cfg_rh)  # this sets the variable(s): self.coarse_net, self.refine_net
        VPoser_Loader.__init__(self, cfg_vp)   # this sets the variable(s): self.mime_net
        self.device = f'cuda:{cfg_vp.cuda_id}'
        self.cfg = OmegaConf.structured({**cfg_rh, **cfg_vp})


    def get_obstacle_info(self, obstacles_dict):
       """
       From obstacle_list with obstacle name, position and orientation, get vertices and normals from Mesh.

       :param obstacles_dict        (dict) with keys containing obstacle names and values list of [verts, faces]

       :return obstacles_info       (list) containing dicts with keys ['o_verts', 'o_faces'] - each a torch.Tensor.
       """
       obstacles_info = []
       for _, [verts, faces] in obstacles_dict.items():
            obj_verts = torch.from_numpy(verts.astype('float32'))  # (N, 3)
            obj_faces = torch.LongTensor(faces.astype('float32'))  # (F, 3)
            obstacles_info.append({'o_verts': obj_verts, 'o_faces': obj_faces})
       return obstacles_info


    def perform_optim(self, z_init, transl_init, global_orient_init, w_init, a_init, obstacles_dict,
                      obj_bps, object_mesh, obj_transl, obj_global_orient, obj_name,
                      model_name='flex'):
        """
        Main controller for optimization across 4 parameters.

        :param z_init                (torch.Tensor) -- size (b, 32) on device
        :param transl_init           (torch.Tensor) -- size (b, 3) on device
        :param global_orient_init    (torch.Tensor) -- size (b, 3) on device
        :param w_init                (torch.Tensor) -- size (b, 16) on device
        :param a_init                (torch.Tensor) -- size (b, 3) on device
        :param obstacles_dict        (dict) with keys containing obstacle names and values list of [verts, faces]
        :param obj_bps               (torch.Tensor) -- size (1, 4096) object bps representation for grasping object
        :param obj_transl            (torch.Tensor) -- size (1, 3) object translation for grasping object
        :param obj_global_orient     (torch.Tensor) -- size (1, 3) object global orientation for grasping object
        :param model_name            (str) -- Model class. Either 'flex' or 'latent'

        :return curr_res             (dict) of ['pose_init', 'transl_init',  'global_orient_init', 'pose_final', 'transl_final', 'global_orient_final', 'rh_verts, 'loss_dict', 'losses']
        """
        bs = self.cfg.batch_size
        pose_init = self.mime_net.decode(z_init)['pose_body'].reshape(bs, -1).detach().cpu()  # save for return

        # Save obstacle info (vertices, normals) required for loss computation.
        obstacles_info = self.get_obstacle_info(obstacles_dict)

        # Define if optimization is over transl and/or global orient.
        extras = {'obj_transl': obj_transl, 'obj_global_orient': obj_global_orient,
                  'obstacles_info': obstacles_info, 'obj_name': obj_name,
                  'object_mesh': object_mesh, 'bps_obj': obj_bps}

        # (*) Adapt model to this example.
        out_optim = optimize_findz(
                            cfg=self.cfg,
                            gan_body=self.mime_net,
                            gan_rh=[self.coarse_net, self.refine_net],
                            z_init=z_init,
                            transl_init=transl_init,
                            global_orient_init=global_orient_init,
                            w_init=w_init,
                            a_init=a_init,
                            num_iterations=self.cfg.n_iter,
                            display=True,
                            extras=extras,
                            model_name=model_name
                        )

        dout, loss_dict, losses = out_optim
        dout = {k: dout[k].detach() for k in dout.keys()}

        # Return loss and result dict.
        curr_res = {'pose_init': pose_init,          'transl_init': transl_init,     'global_orient_init': global_orient_init,
                    'pose_final': dout['pose_body'], 'transl_final': dout['transl'], 'global_orient_final': dout['global_orient'],
                    'rh_verts': dout['rh_verts'], 'z_final': dout['z'], 'human_vertices': dout['human_vertices']}
        curr_res = {k: v.detach().cpu() for k,v in curr_res.items()}
        curr_res['loss_dict'] = loss_dict
        curr_res['losses'] = losses  # NOTE: losses is a dict of lists.

        return curr_res


    def displace(self, obj, test_pose, bs, object_mesh):
        """
        Given a new test-pose, displace the object.

        :param obj                  (str)
        :param test_pose            (list)
        :param bs                   (int)
        :param object_mesh          (list) -- [obj_verts, obj_faces] loaded directly from Mesh file

        :return bps_object          (torch.Tensor on device) -- (bs, 4096)     - bps representation of object in distribution for grasping (useful for GrabNet, not penetration losses)
        :return bps_object_verts    (torch.Tensor on device) -- (bs, 10000, 3) - vertex subset corresponding to above BPS representation   (useful for GrabNet, not penetration losses)
        :return rand_rotdeg         (torch.Tensor on device) -- (bs, 3)        - random rotation degrees applied to get BPS
        """
        # Displacement - Move object in a new translation & orientation (GRAB frame-of-reference).
        din = {}
        din['obj_tran'] = test_pose[0].to(self.device)
        din['obj_glob'] = test_pose[1].to(self.device)

        # Load object verts, bps, etc from the same distribution as training (random rotation).
        rand_rotdeg = torch.rand([bs, 3]) * 360                                                                 # (bs, 3)
        rnd_rotmat = euler_torch(rand_rotdeg)                                                                   # (bs, 3, 3)
        bps_object_verts, _ = load_obj_verts(rnd_rotmat, object_mesh)
        bps_object_verts = bps_object_verts.to(self.device)                                                     # (bs, 10000, 3)
        bps = bps_torch(custom_basis = torch.from_numpy(np.load(self.cfg.bps_pth)['basis']).to(self.device))
        bps_object = bps.encode(bps_object_verts, feature_type='dists')['dists']                                # (bs, 4096)
        rand_rotdeg = rand_rotdeg.to(self.device)

        # Return.
        return bps_object, bps_object_verts, rand_rotdeg


    def get_inits(self, test_pose, obj_bps, bs):
        """
        :param test_pose      (list) of 2 torch.Tensors of size (1,3) & (1,3) for transl and global_orient of object respectively
        :param obj_bps        (torch.Tensor) -- (bs, 4096)     - bps representation of object in distribution for grasping (useful for GrabNet, not penetration losses)
        :param bs             (int) - batch size - product of number of initializations for each variable

        :return t_inits       (torch.Tensor) on device (bs, 3)
        :return g_inits       (torch.Tensor) on device (bs, 3)
        :return z_inits       (torch.Tensor) on device (bs, 32)
        :return w_inits       (torch.Tensor) on device (bs, 16)
        """
        t_inits = (test_pose[0] + torch.rand(bs, 3) * 0.5).to(self.device)
        z_inits = torch.zeros(bs, 32).to(self.device)
        g_inits = recompose_angle(torch.rand(bs) * math.pi, torch.zeros(bs), torch.ones(bs) * 1.5, 'aa').to(self.device)
        w_inits = self.coarse_net.sample_poses(obj_bps)['z']
        return t_inits, g_inits, z_inits, w_inits


    def optimize(self, obj_name, test_pose, obstacles_dict, bs=1, model_name='flex'):
        """Given an object, perform grid search to optimize over 4 variables: pelvis translation, pelvis global orientation, body pose and latent (z) of VPoser.
        Save the top-k results which have the lowest loss based on constraints specified in loss function in `optimize_findz`.

        :param obj_name       (str)
        :param test_pose      (list) of 2 torch.Tensors of size (1,3) & (1,3) for transl and global_orient of object respectively
        :param obstacles_dict (dict) with keys containing obstacle names and values list of [verts, faces]
        :param bs             (int) - batch size
        :param model_name     (str) - model class. Either 'flex' or 'latent'

        :return results       (dict) - keys ['pose_init', 'transl_init', 'global_orient_init', 'pose_final', 'transl_final', 'global_orient_final', 'rh_verts', 'loss_dict', 'losses']
                                       where each is a tensor of size (topk, ..) except `loss_dict` which is itself a dict of detailed losses.
        """
        # (*) Set generative model(s) to eval mode.
        self.mime_net.eval()
        self.coarse_net.eval()
        self.refine_net.eval()

        object_mesh = Mesh(filename=os.path.join(self.cfg.obj_meshes_dir, obj_name + '.ply'), vscale=1.)
        object_mesh.reset_normals()
        object_mesh = [object_mesh.v, object_mesh.f]

        # (*) Setup (heuristic) initialization for optimization parameters.
        obj_bps, _, a_inits = self.displace(obj_name, test_pose, bs, object_mesh)
        # NOTE: for above
        #   - a_inits stores the random rotation that we used to get obj_bps.
        #   - We will use a_inits later to tranform the predicted hand pose back in the correct coordinate system
        t_inits, g_inits, z_inits, w_inits = self.get_inits(test_pose, obj_bps, bs)

        # (*) Perform main optimization for 4 initializations.
        start_time = time.time()
        obj_transl, obj_global_orient = test_pose[0].to(self.device), test_pose[1].to(self.device)
        curr_res = self.perform_optim(z_inits, t_inits, g_inits, w_inits, a_inits,
                                      obstacles_dict, obj_bps, object_mesh,
                                      obj_transl, obj_global_orient, obj_name, model_name)

        # (*) Save topk results.
        results = replace_topk(curr_res, self.cfg.topk)
        print("--- %s seconds ---" % (time.time() - start_time))

        return results


def main():
    # === Add command line arguments.
    parser = argparse.ArgumentParser(description='Test-Time Optimization for Full-Body Human Pose Search')
    parser.add_argument('--cuda_id', default=0, type=int)
    parser.add_argument('--exp_cfg', default='flex/configs/flex.yaml', type=str, help='The default config path for this project.')
    parser.add_argument('--rh_cfg', default='flex/configs/rh.yaml', type=str, help='The default config path for right-hand grasping model.')
    parser.add_argument('--gender', default='neutral', type=str, help='The gender for which the visualization should be generated.')
    parser.add_argument('--save_pth', default='save', type=str, help='The path where to save the result.')
    parser.add_argument('--obj_name', type=str, help='Name of the object.')
    parser.add_argument('--receptacle_name', type=str, help='Name of the receptacle.')
    parser.add_argument('--ornt_name', type=str, help='Orientation -- all or up.')
    parser.add_argument('--index', default=0, type=int, help='0/1/2')
    args = parser.parse_args()

    # === Configuration.
    cfg_vp = OmegaConf.structured(Config)                   # Load base config
    cfg_yaml = OmegaConf.load(vars(args).pop('exp_cfg'))    # Load yaml config
    cfg_rh = OmegaConf.load(vars(args).pop('rh_cfg'))       # Create config for right-hand model
    # Extract excess variables from args.
    cfg_rh.cuda_id = args.cuda_id
    obj_name = vars(args).pop('obj_name')
    receptacle_name = vars(args).pop('receptacle_name')
    ornt_name = vars(args).pop('ornt_name')
    index = vars(args).pop('index')
    save_pth = vars(args).pop('save_pth')
    gender = vars(args).pop('gender')
    # Combine base, yaml and cmd configs to get final vposer config
    cfg_vp = OmegaConf.merge(cfg_vp, cfg_yaml, vars(args))
    # Overwrite with excess as required.
    cfg_rh.batch_size = cfg_vp.bs
    cfg_vp.gender = gender

    # === Check if already computed.
    print(f'-------- Processing OBJ: {obj_name}, RECEPT: {receptacle_name}, GENDER: {cfg_vp.gender} -------------')
    path_save = f'{save_pth}/{obj_name}/{receptacle_name}/{ornt_name}_{index}.npz'
    if os.path.exists(path_save):
        print(f'Already computed for {path_save}')
        return
    os.makedirs(f'{save_pth}/{obj_name}/{receptacle_name}', exist_ok=True)

    # === Main optimization instance.
    tto = Optimize(cfg_rh=cfg_rh, cfg_vp=cfg_vp)

    # === Load saved data from file.
    recept_dict = dict(np.load('data/replicagrasp/receptacles.npz', allow_pickle=1))
    dset_info_dict = dict(np.load('data/replicagrasp/dset_info.npz', allow_pickle=1))
    transl_grab, orient_grab, recept_idx = dset_info_dict[f'{obj_name}_{receptacle_name}_{ornt_name}_{index}']
    pose = [torch.Tensor(transl_grab), rotmat2aa(torch.Tensor(orient_grab).reshape(1,1,1,9)).reshape(1,3)]
    recept_v, recept_f = recept_dict[receptacle_name][recept_idx][0], recept_dict[receptacle_name][recept_idx][1]
    obstacles_dict = {receptacle_name: [recept_v, recept_f]}

    # === Run search given an input object, it's pose and obstacles info.
    res = tto.optimize(obj_name=obj_name, test_pose=pose, obstacles_dict=obstacles_dict, bs=cfg_vp.bs, model_name=cfg_vp.model_name)
    final_results = []
    for i in range(cfg_vp.topk):
        final_results.append({
            'human_vertices': res['human_vertices'][i].detach().cpu().numpy(),
            'pose': res['pose_final'][i].reshape(21, 3).detach().cpu().numpy(),
            'transl': res['transl_final'][i].detach().cpu().numpy(),
            'global_orient': aa2rotmat(res['global_orient_final'])[i].view(3, 3).detach().cpu().numpy(),
            'rh_verts': res['rh_verts'][i].detach().cpu().numpy(),
            'z_final': res['z_final'][i].detach().cpu().numpy(),
        })

    # === Save results.
    os.makedirs(save_pth, exist_ok=True)
    np.savez(path_save, {'final_results': final_results, 'cfg_vp': cfg_vp,
                         'args': args, 'datetime': str(datetime.now())})


if __name__ == "__main__":
    main()