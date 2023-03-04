"""
Main functions for test-time optimization.
"""
from flex.tools.utils import point2point_signed, aa2rotmat, local2global, load_obj_verts
from flex.tools.registry import registry
from flex.models.pgp_model import *  # this should come after registry import
from flex.tools.utils import recompose_angle
from flex.tools.utils import euler_torch
from flex.tools import pytorch_geometric

from torch.nn import functional as F
from bps_torch.bps import bps_torch
from collections import Counter
from functools import partial
from tqdm.auto import tqdm
import scipy.sparse as sp
import torch.nn as nn
import numpy as np
import kaolin
import torch
import smplx
import mano

# ---- Main loss functions.
class Losses(object):
    def __init__(
        self,
        cfg,
        gan_body,
        gan_rh
    ):
        # Preliminaries.
        self.cfg = cfg
        self.device = 'cuda:'+str(cfg.cuda_id) if cfg.cuda_id != -1 else 'cpu'
        self.gan_body = gan_body
        self.gan_rh_coarse, self.gan_rh_refine = gan_rh

        # Body model.
        self.sbj_m = smplx.create(model_path=cfg.smplx_dir,
                                  model_type='smplx',
                                  gender=cfg.gender,
                                  use_pca=False,
                                  flat_hand_mean=True,
                                  batch_size=cfg.batch_size).to(self.device).eval()
        # Hand model.
        self.rh_m = mano.load(model_path=cfg.mano_dir,
                              is_right=True,
                              model_type='mano',
                              gender=cfg.gender,
                              num_pca_comps=45,
                              flat_hand_mean=True,
                              batch_size=cfg.batch_size).to(self.device).eval()

        # Misc.
        self.sbj_verts_region_map = np.load(self.cfg.sbj_verts_region_map_pth, allow_pickle=True)  # (10475,)
        self.adj_matrix_original = np.load(self.cfg.adj_matrix_orig)
        if cfg.subsample_sbj:
            self.sbj_verts_id = np.load(self.cfg.sbj_verts_simplified)                             # (625,)
            self.sbj_faces_simplified = np.load(self.cfg.sbj_faces_simplified)
            self.sbj_verts_region_map = self.sbj_verts_region_map[self.sbj_verts_id]
            self.adj_matrix_simplified = np.load(self.cfg.adj_matrix_simplified)
        with open(cfg.mano2smplx_verts_ids, 'rb') as f: self.correspondences = np.load(f)          # (778,)
        self.interesting = dict(np.load(cfg.interesting_pth))                                      # dict of ['interesting_vertices', 'interesting_vertices_larger']


    def get_rh_match_loss(self, z, transl, global_orient, w, angle, extras):
        """
        Get full-body of human based on pose and parameters (transl, global_orient), along with right-hand pose obtained from rh-grasping model.
        Compute loss between predicted joint/vertex locations versus expected for right hand.

        :param z                (torch.Tensor)
        :param transl           (torch.Tensor)
        :param global_orient    (torch.Tensor)
        :param w                (torch.Tensor)
        :param angle            (torch.Tensor)
        :param extras           (dict) - keys ['obj_bps, 'obj_bps_verts', 'obj_transl'] (stuff that is not optimized over and is necessary for loss computation)

        :return rh_match_loss   (torch.Tensor) - (bs,)
        :return bm_output       (SMPLX body-model output)
        :return rv              (torch.Tensor) - (bs, 778, 3)
        :return rf              (torch.Tensor) - (1538, 3)
        """
        bs = z.shape[0]

        # 1. Forward pass for body-gan to get body-pose.
        sbj_pose = self.gan_body.decode(z)['pose_body'].reshape(bs,-1)                                                    # (b, 63)
        angle_rotmat = euler_torch(360*angle)
        obj_bps_verts, _ = load_obj_verts(angle_rotmat, extras['object_mesh'])
        obj_bps_verts = obj_bps_verts.to(self.device)
        bps = bps_torch(custom_basis=torch.from_numpy(np.load(self.cfg.bps_pth)['basis']).to(self.device))
        bps_object = bps.encode(obj_bps_verts, feature_type='dists')['dists']

        # 2. Forward pass for rh-gan & transformation to get rh-pose and joints/vertices constraint.
        drec_cnet = self.gan_rh_coarse.decode(w, bps_object)
        verts_rh_gen_cnet = self.rh_m(**drec_cnet).vertices                                                               # (b, 778, 3)
        _, h2o = point2point_signed(verts_rh_gen_cnet, obj_bps_verts.to(self.device))                                     # (b, 778)

        drec_cnet['trans_rhand_f'] = drec_cnet['transl']                                                                  # (b, 3)
        drec_cnet['global_orient_rhand_rotmat_f'] = aa2rotmat(drec_cnet['global_orient']).view(-1, 3, 3)                  # (b, 3, 3)
        drec_cnet['fpose_rhand_rotmat_f'] = aa2rotmat(drec_cnet['hand_pose']).view(-1, 15, 3, 3)                          # (b, 15, 3, 3)
        drec_cnet['verts_object'] = obj_bps_verts.to(self.device)                                                         # (b, N, 3)
        drec_cnet['h2o_dist']= h2o.abs()                                                                                  # (b, 778)
        drec_rnet = self.gan_rh_refine(**drec_cnet)                                                                       # dict of ['global_orient', 'hand_pose', 'transl', 'fullpose']
        rh_out = self.rh_m(**drec_rnet)
        joints_constraint = rh_out.joints                                                                                 # (b, 16, 3)
        vertices_constraint = rh_out.vertices                                                                             # (b, 778, 3)

        # 3. Transform appropriately.
        obj_global_orient_rotmat = aa2rotmat(extras['obj_global_orient'].reshape(1,1,1,3)).reshape(1,3,3).repeat(bs,1,1)  # (bs, 3, 3)
        R = torch.bmm(obj_global_orient_rotmat, torch.inverse(angle_rotmat)).to(self.device)                              # (b, 3, 3)
        t = extras['obj_transl'].repeat(bs,1)                                                                             # (b, 3)
        joints_constraint = local2global(joints_constraint, R, t)                                                         # (b, 16, 3)
        vertices_constraint = local2global(vertices_constraint, R, t)                                                     # (b, 778, 3)

        # 4. Full-body generation using steps 1 & 2 & 3.
        bodydict_output = {'body_pose': sbj_pose, 'transl': transl, 'global_orient': global_orient, 'right_hand_pose': drec_rnet['hand_pose']}
        bm_output = self.sbj_m(**bodydict_output)
        # Make the lowest point of the human touch the ground
        bm_output.vertices[..., 2] -= bm_output.vertices[..., 2].min(-1, keepdims=True)[0]

        # 5. Loss computation.
        if self.cfg.rh_match_type == 'joints':
            joints_output_wrist = bm_output.joints[:, 21:22, :]
            joint_output_fingers = bm_output.joints[:, 40:55, :]
            joints_output = torch.cat((joints_output_wrist, joint_output_fingers), 1).to(self.device)
            match_loss = F.mse_loss(joints_output, joints_constraint, reduction='none')                                       # (b, 16, 3)
            joints_wts = torch.Tensor([[10,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3]]).repeat(bs,1).reshape(bs,16,1).to(self.device)     # (b, 16, 1) - Weigh wrist and fingertips higher than middle, higher than start, etc.
            match_loss *= joints_wts

            # 6. Average
            match_loss = match_loss.reshape(bs, -1).mean(1)
        else:  # verts
            bm_output_vertices = bm_output.vertices[:, self.correspondences]
            match_loss = F.mse_loss(bm_output_vertices, vertices_constraint, reduction='none').reshape(bs, -1).mean(1)
            if self.cfg.alpha_interesting_verts > 0:
                iv = self.interesting['interesting_vertices']
                bm_output_vertices_interest = bm_output_vertices[:, iv]
                vertices_constraint_interest = vertices_constraint[:, iv]
                match_loss_interest = F.mse_loss(bm_output_vertices_interest, vertices_constraint_interest,
                                                 reduction='none').reshape(bs, -1).mean(1)
                match_loss += self.cfg.alpha_interesting_verts * match_loss_interest

        return match_loss, bm_output, vertices_constraint, torch.from_numpy(self.rh_m.faces.astype(float)).to(self.device)


    def intersection(self, sbj_verts, obj_verts, sbj_faces, obj_faces, full_body=True, adjacency_matrix=None):
        """
        Compute intersection penalty between body and object (or obstacle) given vertices and normals of both.

        :param sbj_verts                (torch.Tensor) on device - (bs, N_sbj, 3)
        :param obj_verts                (torch.Tensor) on device - (1, N_obj, 3)
        :param sbj_faces                (torch.Tensor) on device - (F_sbj, 3)
        :param obj_faces                (torch.Tensor) on device - (F_obj, 3)
        :param full_body                (bool) -- for full-body if True; else for rhand
        :param adjacency_matrix         (optional)

        :return penet_loss_batched_in   (torch.Tensor) - (bs,) - loss values for each batch element - penetration
        :return penet_loss_batched_out  (torch.Tensor) - (bs,) - loss values for each batch element - outside
        """
        device = sbj_verts.device
        bs = sbj_verts.shape[0]
        obj_verts = obj_verts.repeat(bs, 1, 1)                                                                               # (bs, N_obj, 3)
        num_obj_verts, num_sbj_verts = obj_verts.shape[1], sbj_verts.shape[1]
        penet_loss_batched_in, penet_loss_batched_out = torch.zeros(bs).to(device), torch.zeros(bs).to(device)
        thresh = self.cfg.intersection_thresh

        # (*) Object to subject.
        if self.cfg.obstacle_obj2sbj:
            # 1. Use Kaolin to calculate sign (True if inside, False if outside)
            sign = kaolin.ops.mesh.check_sign(sbj_verts, sbj_faces, obj_verts)                                               # (bs, N_obj)
            ones = torch.ones_like(sign.long())                                                                              # (bs, N_obj)
            # 2. Negative for penetration, Positive for outside (to keep consistent with previous format).
            sign = torch.where(sign, -ones, ones)                                                                            # (bs, N_obj)
            # 3. Calculate absolute distance of points from mesh, and multiply by sign.
            face_vertices = kaolin.ops.mesh.index_vertices_by_faces(sbj_verts, sbj_faces)                                    # (bs, F_sbj, 3, 3)
            dist, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(obj_verts.contiguous(), face_vertices)           # (bs, N_obj)
            obj2sbj = dist * sign                                                                                            # (bs, N_obj)
            # 4. Average across batch for negative and positive values.
            zeros_o2s, ones_o2s = torch.zeros_like(obj2sbj).to(device), torch.ones_like(obj2sbj).to(device)
            loss_o2s_in = torch.sum(abs(torch.where(obj2sbj<thresh, obj2sbj-thresh, zeros_o2s)), 1) / num_obj_verts          # (bs,) -- averaged across (bs, N_obj)
            loss_o2s_out = torch.sum(torch.log(torch.where(obj2sbj>thresh, obj2sbj+ones_o2s, ones_o2s)), 1) / num_obj_verts  # (bs,) -- averaged across (bs, N_obj)
            # 5. Add to final loss.
            penet_loss_batched_in += loss_o2s_in
            penet_loss_batched_out += loss_o2s_out

        # (*) Subject to object.
        if self.cfg.obstacle_sbj2obj:
            # 0. Simplify obstacle faces - many have determinant ~0, i.e., it is a degenerate triangle.
            # NOTE: No need to do for all elements in batch because faces are the same, so just repeat.
            face_vertices = kaolin.ops.mesh.index_vertices_by_faces(obj_verts, obj_faces)                                    # (bs, F_obj, 3, 3)
            indices_good_faces = (face_vertices[0].det().abs() > 0.001)                                                      # (F_obj)
            obj_faces = obj_faces[indices_good_faces]
            face_vertices = face_vertices[0][indices_good_faces][None].repeat(bs, 1, 1, 1)                                   # (bs, F_obj_good, 3, 3)
            # 1. Use Kaolin to calculate sign (True if inside, False if outside)
            sign = kaolin.ops.mesh.check_sign(obj_verts, obj_faces, sbj_verts)                                               # (bs, N_sbj)
            ones = torch.ones_like(sign.long())                                                                              # (bs, N_sbj)
            # 2. Negative for penetration, Positive for outside (to keep consistent with previous format).
            sign = torch.where(sign, -ones, ones)                                                                            # (bs, N_sbj)
            # 3. Calculate absolute distance of points from mesh, and multiply by sign.
            dist, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(sbj_verts.contiguous(), face_vertices)           # (bs, N_sbj)
            sbj2obj = dist * sign                                                                                            # (bs, N_sbj)
            # 4. Average across batch for negative and positive values.
            zeros_s2o, ones_s2o = torch.zeros_like(sbj2obj).to(device), torch.ones_like(sbj2obj).to(device)
            loss_s2o_out = torch.sum(torch.log(torch.where(sbj2obj>thresh, sbj2obj+ones_s2o, ones_s2o)), 1) / num_sbj_verts  # (bs,)  -- averaged across (bs, N_sbj)
            # 4.1. Special case for sbj2obj negative values - check whether to do connected components or not.
            loss_s2o_in = torch.sum(abs(torch.where(sbj2obj<thresh, sbj2obj-thresh, zeros_s2o)), 1) / num_sbj_verts          # (bs,)  -- averaged across (bs, N_sbj)
            if full_body and self.cfg.obstacle_sbj2obj_extra == 'connected_components' and loss_s2o_in.mean() > 0:
                # Connected components based loss.
                edges = np.stack(np.where(adjacency_matrix))
                num_nodes = adjacency_matrix.shape[0]
                v_to_edges = torch.zeros((num_nodes, edges.shape[1]))
                v_to_edges[edges[0], range(edges.shape[1])] = 1
                v_to_edges[edges[1], range(edges.shape[1])] = 1

                indices_inter = (sbj2obj < thresh)
                v_to_edges = v_to_edges[None].expand(bs, -1, -1).clone()
                v_to_edges[torch.where(indices_inter)] = 0
                edges_indices = v_to_edges.sum(1) == 2
                num_inter_v = indices_inter.sum(-1)

                for i in range(bs):
                    if loss_s2o_in[i] > 0:
                        edges_i = edges[:, edges_indices[i]]
                        adj = pytorch_geometric.to_scipy_sparse_matrix(edges_i, num_nodes=num_nodes)
                        n_components, labels = sp.csgraph.connected_components(adj)

                        n_components -= num_inter_v[i]  # Inside obstacles are not taken into account
                        if n_components > 1:
                            indices_out = torch.ones([num_sbj_verts])
                            indices_out[indices_inter[i]] = 0
                            labels_ = labels[indices_out.bool()]
                            # We penalize only the vertices that are out, but the penalization is wrt the original
                            # edge, not including the threshold.
                            most_common_label = Counter(labels_).most_common()[0][0]
                            penalized_joints = (labels != most_common_label) * indices_out.bool().numpy()
                            loss_s2o_in[i] += sbj2obj[i][penalized_joints].sum() / num_sbj_verts

            # 5. Add to final loss.
            penet_loss_batched_in += loss_s2o_in
            penet_loss_batched_out += loss_s2o_out

        # (*) Return final.
        return penet_loss_batched_in, penet_loss_batched_out


    def get_rh_obstacle_penet_loss(self, rv, rf, extras):
        """
        Compute penetration loss between right-hand grasp and all provided obstacle vertices.

        :param rv                           (torch.Tensor) - (bs, 778, 3)
        :param rf                           (torch.Tensor) - (bs, 1538, 3)
        :param extras                       (dict)         - keys ['ov', 'obj_normals', 'o_verts_wts'] (stuff that is not optimized over and that can be copied over from GT and is necessary for loss computation)

        :return obstacle_loss_batched_in   (torch.tensor) - (bs,)
        :return obstacle_loss_batched_out  (torch.tensor) - (bs,)
        """
        bs = self.cfg.batch_size
        obstacle_loss_batched_in, obstacle_loss_batched_out = torch.zeros(bs).to(self.device), torch.zeros(bs).to(self.device)
        for obstacle in extras['obstacles_info']:
            olb_in, olb_out = self.intersection(rv, obstacle['o_verts'][None].to(self.device), rf, obstacle['o_faces'].to(self.device), False)
            obstacle_loss_batched_in += olb_in
            obstacle_loss_batched_out += olb_out
        if len(extras['obstacles_info']):
            obstacle_loss_batched_in /= len(extras['obstacles_info'])
            obstacle_loss_batched_out /= len(extras['obstacles_info'])
        return obstacle_loss_batched_in, obstacle_loss_batched_out


    def get_obstacle_penet_loss(self, bm_output, extras):
        """
        Compute penetration loss between human and all provided obstacle vertices.

        :param bm_output                    (SMPLX body-model output)
        :param extras                       (dict)         - keys ['o_verts', 'o_faces'] (stuff that is not optimized over and that can be copied over from GT and is necessary for loss computation)

        :return obstacle_loss_batched_in   (torch.tensor) - (bs,)
        :return obstacle_loss_batched_out  (torch.tensor) - (bs,)
        """
        # Preliminaries.
        bs = self.cfg.batch_size

        # Load subject vertices and faces.
        bv = bm_output.vertices.reshape(bs,-1,3)                                                # (bs, 10475, 3)
        bf = torch.LongTensor(self.sbj_m.faces.astype('float32')).to(self.device)               # (20908, 3)
        if self.cfg.subsample_sbj:
            bf = self.sbj_faces_simplified                                                      # (1269, 3)
            bv = bv[:, self.sbj_verts_id, :]                                                    # (bs, 625, 3)
            adjacency_matrix = self.adj_matrix_simplified
        else:
            adjacency_matrix = self.adj_matrix_original

        # Compute loss for each obstacle.
        obstacle_loss_batched_in, obstacle_loss_batched_out = torch.zeros(bs).to(self.device), torch.zeros(bs).to(self.device)
        for obstacle in extras['obstacles_info']:
            olb_in, olb_out = self.intersection(bv, obstacle['o_verts'][None].to(self.device), bf, obstacle['o_faces'].to(self.device), True, adjacency_matrix)
            obstacle_loss_batched_in += olb_in
            obstacle_loss_batched_out += olb_out
        if len(extras['obstacles_info']):
            obstacle_loss_batched_in /= len(extras['obstacles_info'])
            obstacle_loss_batched_out /= len(extras['obstacles_info'])
        return obstacle_loss_batched_in, obstacle_loss_batched_out


    def get_lowermost_loss(self, bm_output):
        """
        Compute absolute distance between lowermost point of body. This assumes that floor is at zero height.
        Useful to correct for when ground-loss is not perfect.

        :param bm_output       (SMPLX body-model output)

        :return lowermost_loss (torch.tensor) - (bs,)
        """
        bs = self.cfg.batch_size
        body_vertices = bm_output.vertices.reshape(bs,-1,3)                # (bs, 10475, 3)
        lowermost_loss = abs(torch.min(body_vertices[:, :, 2], 1).values)  # absolute distance of lowest vertex location on body (averaged for batch)
        return lowermost_loss


    def get_gaze_loss(self, bm_output, extras):
        """
        Compute gaze angle between head vector and back-of-the-head-to-object vectors.

        :param bm_output       (SMPLX body-model output)
        :param extras          (dict)         - keys containing 'obj_transl'

        :return gaze_loss (torch.tensor) - (bs,)
        """
        bs = self.cfg.batch_size
        body_vertices = bm_output.vertices.reshape(bs,-1,3)                             # (bs, 10475, 3)
        head_front, head_back = body_vertices[:, 8970], body_vertices[:, 8973]          # (bs, 3); (bs, 3)
        obj_transl = extras['obj_transl'].repeat(bs, 1)                                 # (bs, 3)
        # 1. Get vectors.
        vec_head, vec_obj = head_front - head_back, obj_transl - head_back              # (b, 3); (b, 3)
        # 2. Dot product.
        dot = torch.bmm(vec_head.view(bs, 1, -1), vec_obj.view(bs, -1, 1))[:, 0, 0]     # (b,)
        # 3. Magnitude.
        norm_head = torch.norm(vec_head, dim=1) + 1e-4                                  # (b,)
        norm_obj = torch.norm(vec_obj, dim=1)+ 1e-4                                     # (b,)
        # 4. Get angle using above steps.
        gaze_loss = torch.arccos(dot / (norm_head * norm_obj))
        return gaze_loss


    def get_wrist_loss(self, bm_output, rh_vertices):
        """
        Compute wrist angle between MANO grasp hand and SMLPX full-body hand.

        :param bm_output       (SMPLX body-model output)
        :param rh_vertices     (MANO hand-model output vertices)
        :param correspondences (SMLPX to MANO correspondences)

        :return gaze_loss (torch.tensor) - (bs,)
        """
        bs = self.cfg.batch_size
        body_vertices = bm_output.vertices.reshape(bs,-1,3)                             # (bs, 10475, 3)
        rh_left, rh_right = body_vertices[:, self.correspondences][:, 90], \
                            body_vertices[:, self.correspondences][:, 51]               # (bs, 3); (bs, 3)
        body_left, body_right = rh_vertices[:, 90], rh_vertices[:, 51]                  # (bs, 3); (bs, 3)
        # 1. Get vectors.
        vec_rh, vec_bm = rh_left - rh_right, body_left - body_right                     # (b, 3); (b, 3)
        # 2. Dot product.
        dot = torch.bmm(vec_rh.view(bs, 1, -1), vec_bm.view(bs, -1, 1))[:, 0, 0]        # (b,)
        # 3. Magnitude.
        norm_rh = torch.norm(vec_rh, dim=1) + 1e-4                                      # (b,)
        norm_bm = torch.norm(vec_bm, dim=1) + 1e-4                                      # (b,)
        # 4. Get angle using above steps.
        wrist_loss = torch.arccos(dot / (norm_rh * norm_bm))
        return wrist_loss


    def gan_loss(self, z, transl, global_orient, w, a, extras={}, alpha_lowermost=0.0, alpha_rh_match=1.0,
                 alpha_obstacle_in=0.0, alpha_obstacle_out=0.0, alpha_gaze=0.0,
                 alpha_rh_obstacle_in=0.0, alpha_rh_obstacle_out=0.0, alpha_wrist=0.0):
        """
        GAN loss for Test-Time Optimization:
            (1) MSE loss for full-body joints/vertices after passing body parameters of VPoser through in SMPLX.
            (2) Lowermost loss which makes lowest body vertex along vertical axis zero
            (3) Human-object penetration loss.
            (4) Human-obstacle penetration loss.

        :param z             (torch.Tensor) - (1, 32)
        :param transl        (list of 3 torch.Tensors) - each of shape (1,)
        :param global_orient (list of 3 torch.Tensors) - each of shape (1,)
        :param w             (torch.Tensor) - (1, 16)
        :param a             (torch.Tensor) - (1, 3, 3)
        :param extras        (dict)         - keys ['ov', 'obj_normals'] (stuff that is not optimized over and that can be copied over from GT and is necessary for loss computation)

        :return loss_dict    (dict of torch.Tensor) - each single loss value
        """
        # Preliminaries: Combine parameter parts.

        transl_x, transl_y, transl_z = transl
        transl = torch.stack([transl_x, transl_y, transl_z], 1)
        global_orient1, global_orient2, global_orient3 = global_orient
        if self.cfg.orient_optim_type == 'aa':
            # aa1, aa2, aa3 = global_orient
            global_orient = torch.stack([global_orient1, global_orient2, global_orient3], 1)
        else:  #'ypr'
            # yaw, pitch, roll = global_orient
            global_orient = recompose_angle(global_orient1, global_orient2, global_orient3, 'aa')

        # (*) Joint loss.
        rh_match_loss, bm_output, rv, rf = self.get_rh_match_loss(z, transl, global_orient, w, a, extras)

        # (*) RHand-obstacle(s) penetration loss.
        rh_obstacle_loss_in = rh_obstacle_loss_out = torch.zeros_like(rh_match_loss)
        if alpha_rh_obstacle_in + alpha_rh_obstacle_out > 0:
            rh_obstacle_loss_in, rh_obstacle_loss_out = self.get_rh_obstacle_penet_loss(rv, rf, extras)

        # (*) Human-obstacle(s) penetration loss.
        obstacle_loss_in = obstacle_loss_out = torch.zeros_like(rh_match_loss)
        if alpha_obstacle_in + alpha_obstacle_out > 0:
            obstacle_loss_in, obstacle_loss_out = self.get_obstacle_penet_loss(bm_output, extras)

        # (*) Lowermost Loss.
        lowermost_loss = torch.zeros_like(rh_match_loss)
        if alpha_lowermost > 0:
            lowermost_loss = self.get_lowermost_loss(bm_output)

        # (*) Gaze Loss.
        gaze_loss = torch.zeros_like(rh_match_loss)
        if alpha_gaze > 0:
            gaze_loss = self.get_gaze_loss(bm_output, extras)

        # (*) Wrist Loss.
        wrist_loss = torch.zeros_like(rh_match_loss)
        if alpha_wrist > 0:
            wrist_loss = self.get_wrist_loss(bm_output, rv)

        # Total loss.
        total_loss = lowermost_loss * alpha_lowermost + rh_match_loss * alpha_rh_match + \
                     obstacle_loss_in * alpha_obstacle_in + obstacle_loss_out * alpha_obstacle_out + \
                     rh_obstacle_loss_in * alpha_rh_obstacle_in + rh_obstacle_loss_out * alpha_rh_obstacle_out + \
                     gaze_loss * alpha_gaze + wrist_loss * alpha_wrist
        loss_dict = {'total': total_loss,
                     'lowermost_loss': lowermost_loss,
                     'rh_match_loss': rh_match_loss,
                     'obstacle_loss_in': obstacle_loss_in,
                     'obstacle_loss_out': obstacle_loss_out,
                     'gaze_loss': gaze_loss,
                     'wrist_loss': wrist_loss,
                     'rh_obstacle_loss_in': rh_obstacle_loss_in,
                     'rh_obstacle_loss_out': rh_obstacle_loss_out,
                    }

        return loss_dict


class FLEX(nn.Module):
    def __init__(
        self,
        cfg,
        z_init,
        transl_init,
        global_orient_init,
        w_init,
        a_init,
        gan_body,
        gan_rh,
        task='',
        extras={},
        requires_grad=True
    ):
        super(FLEX, self).__init__()
        self.cfg = cfg
        self.bs = z_init.shape[0]
        self.device = 'cuda:'+str(cfg.cuda_id) if cfg.cuda_id != -1 else 'cpu'
        self.losses = Losses(cfg, gan_body, gan_rh)
        self.gan_body = gan_body
        self.gan_rh_coarse, self.gan_rh_refine = gan_rh
        self.task = task
        self.extras = extras

        # Define Optimization Parameters.
        # (1) Z for Body Pose.
        z_param_init = z_init.detach().clone()
        self.z = nn.Parameter(
            z_param_init, requires_grad=requires_grad
        )

        # (2) Translation.
        transl_param_init = transl_init.detach().clone()
        self.transl_x = nn.Parameter(
            transl_param_init[:, 0], requires_grad=requires_grad
        )
        self.transl_y = nn.Parameter(
            transl_param_init[:, 1], requires_grad=requires_grad
        )
        self.transl_z = nn.Parameter(
            transl_param_init[:, 2], requires_grad=requires_grad
        )

        # (3) Global orientation - either 3 components of axis-angle or (yaw, pitch, roll) resp.
        global_orient_param_init = global_orient_init.detach().clone()
        self.global_orient1 = nn.Parameter(
            global_orient_param_init[:, 0], requires_grad=requires_grad
        )
        self.global_orient2 = nn.Parameter(
            global_orient_param_init[:, 1], requires_grad=requires_grad
        )
        self.global_orient3 = nn.Parameter(
            global_orient_param_init[:, 2], requires_grad=requires_grad
        )

        # (4) W for Hand Pose + Translation + Global orientation.
        w_param_init = w_init.detach().clone()
        self.w = nn.Parameter(
            w_param_init, requires_grad=requires_grad
        )

        # (5) Angle for Object Pose for Hand Grasping.
        a_param_init = a_init.detach().clone()
        self.angle = nn.Parameter(
            a_param_init, requires_grad=requires_grad
        )

        # Don't update generative models.
        for _, param in self.gan_body.named_parameters():
            param.requires_grad = False
        for _, param in self.gan_rh_coarse.named_parameters():
            param.requires_grad = False
        for _, param in self.gan_rh_refine.named_parameters():
            param.requires_grad = False

        # Load pre-trained pose-ground prior model.
        self.pose_ground_prior = registry.get_class('Regress')(cfg).to(self.device)
        self.load_model()
        # Don't update pose-ground prior model.
        for _, param in self.pose_ground_prior.named_parameters():
            param.requires_grad = False


    def load_model(self):
        """
        Load best saved ckpt into self.pose_ground_prior.
        """
        state = torch.load(self.cfg.best_pgprior, map_location=self.device)
        self.pose_ground_prior.load_state_dict(state['state_dict'], strict=True)
        self.pose_ground_prior.eval()  # set to eval mode
        
        vars_pnet = [var[1] for var in self.pose_ground_prior.named_parameters()]
        pnet_n_params = sum(p.numel() for p in vars_pnet if p.requires_grad)
        print('\nTotal Trainable Parameters for Pose-Ground Generative Model (PGP) is %2.2f K.' % (pnet_n_params * 1e-3))
        print('------> Loading PGP model (pre-trained on AMASS) from %s\n' % self.cfg.best_pgprior)


    def pose_ground_pred(self):
        """
        Perform forward pass on pre-trained model for pose-ground relation.
        :return    (dict) of ['transl_z', 'pitch', 'yaw'] -- all torch.Tensors of size (1,)
        """
        with torch.no_grad():
            pose_output = self.gan_body.decode(self.z)['pose_body'].reshape(self.bs, -1)
            pred = self.pose_ground_prior(pose_output)
            return pred


    def get_dout(self, best_transl, best_orient, best_z, best_w, best_angle, extras):
        """
        Use optimization parameters (t, g, z, w) to visualize corresponding mesh results.

        :param best_transl  (torch.Tensor) -- (bs, 3)
        :param best_orient  (torch.Tensor) -- (bs, 3)
        :param best_z       (torch.Tensor) -- (bs, 32)
        :param best_w       (torch.Tensor) -- (bs, 16)
        :param best_angle   (torch.Tensor) -- (bs, 3)

        :return dout        (dict) of keys ['pose_body', 'pose_body_matrot', 'z', 'rh_verts', 'transl', 'global_orient'] -- torch.Tensors of size [(bs, 63); (bs, 21, 9); (bs, 32); (bs, 778, 3); (bs, 3); (bs, 3)]
        """
        # 1. Body pose and latent.
        dout = self.gan_body.decode(best_z)
        dout['pose_body'] = dout['pose_body'].reshape(self.bs, -1)                                                                  # (bs, 63)
        dout['z'] = best_z                                                                                                          # (bs, 32)

        # 2. Rhand final vertices.
        rh_m = self.losses.rh_m
        sbj_m = self.losses.sbj_m

        angle_rotmat = euler_torch(360*best_angle)
        obj_bps_verts, _ = load_obj_verts(angle_rotmat, extras['object_mesh'])
        obj_bps_verts = obj_bps_verts.to(self.device)
        bps = bps_torch(custom_basis=torch.from_numpy(np.load(self.cfg.bps_pth)['basis']).to(self.device))
        bps_object = bps.encode(obj_bps_verts, feature_type='dists')['dists']

        drec_cnet = self.gan_rh_coarse.decode(best_w, bps_object)
        verts_rh_gen_cnet = rh_m(**drec_cnet).vertices                                                                              # (bs, 778, 3)

        _, h2o = point2point_signed(verts_rh_gen_cnet, obj_bps_verts.to(self.device))                                               # (bs, 778)
        drec_cnet['trans_rhand_f'] = drec_cnet['transl']                                                                            # (bs, 3)
        drec_cnet['global_orient_rhand_rotmat_f'] = aa2rotmat(drec_cnet['global_orient']).view(-1, 3, 3)                            # (bs, 3, 3)
        drec_cnet['fpose_rhand_rotmat_f'] = aa2rotmat(drec_cnet['hand_pose']).view(-1, 15, 3, 3)                                    # (bs, 15, 3, 3)
        drec_cnet['verts_object'] = obj_bps_verts.to(self.device)                                                                   # (bs, N, 3)
        drec_cnet['h2o_dist']= h2o.abs()                                                                                            # (bs, 778)
        drec_rnet = self.gan_rh_refine(**drec_cnet)                                                                                 # dict of ['global_orient', 'hand_pose', 'transl', 'fullpose']
        rh_out = rh_m(**drec_rnet)
        vertices_constraint = rh_out.vertices

        obj_global_orient_rotmat = aa2rotmat(self.extras['obj_global_orient'].reshape(1,1,1,3)).reshape(1,3,3).repeat(self.bs,1,1)  # (b, 3, 3)
        R = torch.bmm(obj_global_orient_rotmat, torch.inverse(angle_rotmat)).to(self.device)                                        # (b, 3, 3)
        t = self.extras['obj_transl'].repeat(self.bs,1)                                                                             # (b, 3)
        rh_verts = local2global(vertices_constraint, R, t)                                                                          # (bs, 778, 3)
        dout['rh_verts'] = rh_verts                                                                                                 # (bs, 778, 3)

        # 3. Translation and orientation for full-body.
        dout['transl'] = best_transl                                                                                                # (bs, 3)
        dout['global_orient'] = best_orient                                                                                         # (bs, 3)

        # 4. Full-body generation using steps 1 & 2 & 3.
        bodydict_output = {'body_pose': dout['pose_body'], 'transl': best_transl, 'global_orient': best_orient, 'right_hand_pose': drec_rnet['hand_pose']}
        bm_output = sbj_m(**bodydict_output)
        # Make the lowest point of the human touch the ground
        bm_output.vertices[..., 2] -= bm_output.vertices[..., 2].min(-1, keepdims=True)[0]

        dout['human_vertices'] = bm_output.vertices

        return dout


    def forward(self, mode=None, alpha_lowermost=0.0, alpha_rh_match=1.0, alpha_obstacle_in=0.0, alpha_obstacle_out=0.0, alpha_gaze=0.0, alpha_rh_obstacle_in=0.0, alpha_rh_obstacle_out=0.0, alpha_wrist=0.0):
        """
        :param mode    (str)                -- str indicating which step of n-part optimization is happening - e.g., 'tgz,w'
        :return loss   (torch.Tensor item)  -- for whichever mode is selected
        """
        if self.cfg.pgprior:
            # ---> Use mode to add or remove stuff from optimizer.
            self.z.requires_grad = True if 'z' in mode else False
            self.w.requires_grad = True if 'w' in mode else False
            self.angle.requires_grad = True if 'a' in mode else False
            self.transl_z.requires_grad, self.global_orient2.requires_grad, self.global_orient3.requires_grad = False, False, False        # (z, pitch, roll)
            if 'tg' in mode or 'gt' in mode:
                # Optimize over 3 values (transl_x, transl_y, yaw)
                self.transl_x.requires_grad, self.transl_y.requires_grad, self.global_orient1.requires_grad = True, True, True              # (x, y, yaw)
                # Use `self.z` to get current pose. Pass through pretrained model to get hard constraints for (transl_z, pitch, roll).
                pred_params = self.pose_ground_pred()
                self.transl_z.data, self.global_orient2.data, self.global_orient3.data = pred_params['transl_z'], pred_params['pitch'], pred_params['roll']
            else:
                # Remove all translation & orientation parameters from optimizer.
                self.transl_x.requires_grad, self.transl_y.requires_grad, self.global_orient1.requires_grad = False, False, False           # (x, y, yaw)
        else:
            # ---> Baseline without ground-pose prior
            pass

        loss = self.losses.gan_loss(transl=[self.transl_x, self.transl_y, self.transl_z], global_orient=[self.global_orient1, self.global_orient2, self.global_orient3],
                                    z=self.z, w=self.w, a=self.angle, extras=self.extras,
                                    alpha_lowermost=alpha_lowermost, alpha_rh_match=alpha_rh_match, alpha_obstacle_in=alpha_obstacle_in, alpha_obstacle_out=alpha_obstacle_out, alpha_gaze=alpha_gaze,
                                    alpha_rh_obstacle_in=alpha_rh_obstacle_in, alpha_rh_obstacle_out=alpha_rh_obstacle_out, alpha_wrist=alpha_wrist
                                   )
        return loss



class Latent(FLEX):
    """
    Latent Optimization for smoothing & having a single control over all latents (Reference: https://arxiv.org/abs/2206.09027).
    """
    def __init__(self, *args, **kwargs):
        super(Latent, self).__init__(*args, **kwargs, requires_grad=False)

        # Preliminaries.
        self.z_size = self.z.shape[1]
        self.w_size = self.w.shape[1]
        self.a_size = self.angle.shape[1]
        self.g_size = 3
        self.t_size = 3

        # Create global latent parameters.
        batch_size = self.z.shape[0]
        self.latent_z = nn.Parameter(torch.randn((batch_size, 512), device=self.device), requires_grad=True)
        total_size = self.z_size + self.w_size + self.a_size + self.t_size + self.g_size + 1
        if not self.cfg.pgprior:
            total_size += 3
        mlp = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, total_size))
        self.mlp = mlp.to(self.device)


    def forward(self, mode=None, alpha_lowermost=0.0, alpha_rh_match=1.0, alpha_obstacle_in=0.0,
                alpha_obstacle_out=0.0, alpha_gaze=0.0, alpha_rh_obstacle_in=0.0, alpha_rh_obstacle_out=0.0,
                alpha_wrist=0.0, prediction_scale=None, gradient_scale=None):
        """
        :return loss   (torch.Tensor item)  -- for whichever mode is selected
        """

        prediction_scale = {k: float(v) for k, v in prediction_scale.items()}
        gradient_scale = {k: float(v) for k, v in gradient_scale.items()}

        # (*) ======  Load prediction and set each parameter.
        prediction = self.mlp(self.latent_z)
        d = 0
        if 'z' in mode:
            z = prediction[:, d:d+self.z.shape[1]] * prediction_scale['z']
            d += self.z_size

        else:
            z = self.z

        if 't' in mode:
            transl_x = prediction[:, d] * prediction_scale['transl']  # Initialize in a larger range
            d += 1

            transl_y = prediction[:, d] * prediction_scale['transl']
            d += 1

            if not self.cfg.pgprior:
                transl_z = prediction[:, d] * prediction_scale['transl']
                d += 1

        else:
            transl_x = self.transl_x
            transl_y = self.transl_y
            transl_z = self.transl_z

        if 'g' in mode:
            global_orient1 = prediction[:, d] * prediction_scale['orient']
            d += 1

            if not self.cfg.pgprior:
                global_orient2 = prediction[:, d] * prediction_scale['orient']
                d += 1
                global_orient3 = prediction[:, d] * prediction_scale['orient']
                d += 1

        else:
            global_orient1 = self.global_orient1
            global_orient2 = self.global_orient2
            global_orient3 = self.global_orient3

        if 'w' in mode:
            w = prediction[:, d:d+self.w.shape[1]]
            # Normalize w to make it unit norm -> Sampled from a normal distribution
            w = w / torch.norm(w, dim=1, keepdim=True) * prediction_scale['w']
            d += self.w_size
        else:
            w = self.w

        if 'a' in mode:
            angle = prediction[:, d:d+self.angle.shape[1]] * prediction_scale['angle']
            d += self.a_size
        else:
            angle = self.angle


        # (*) ====== Scale gradients
        # Scale gradients

        if 'z' in mode:
            z = scale_grad(z, gradient_scale['z'])
        if 'tg' in mode or 'gt' in mode:
            transl_x = scale_grad(transl_x, gradient_scale['transl'])
            transl_y = scale_grad(transl_y, gradient_scale['transl'])
            global_orient1 = scale_grad(global_orient1, gradient_scale['orient'])
            # ---
            pose_output = self.gan_body.decode(z)['pose_body'].reshape(self.bs, -1)
            if self.cfg.pgprior:
                pred_params = self.pose_ground_prior(pose_output)
                transl_z, global_orient2, global_orient3 = pred_params['transl_z'], pred_params['pitch'], pred_params['roll']
        if 'w' in mode:
            w = scale_grad(w, gradient_scale['w'])
        if 'a' in mode:
            angle = scale_grad(angle, gradient_scale['angle'])


        # ====== Compute loss.
        loss = self.losses.gan_loss(transl=[transl_x, transl_y, transl_z], global_orient=[global_orient1, global_orient2, global_orient3],
                                    z=z, w=w, a=angle, extras=self.extras,
                                    alpha_lowermost=alpha_lowermost, alpha_rh_match=alpha_rh_match, alpha_obstacle_in=alpha_obstacle_in, alpha_obstacle_out=alpha_obstacle_out, alpha_gaze=alpha_gaze,
                                    alpha_rh_obstacle_in=alpha_rh_obstacle_in, alpha_rh_obstacle_out=alpha_rh_obstacle_out, alpha_wrist=alpha_wrist
                                   )


        # ====== Save and return.
        self.z.data = z
        self.transl_x.data = transl_x
        self.transl_y.data = transl_y
        self.transl_z.data = transl_z
        self.global_orient1.data = global_orient1
        self.global_orient2.data = global_orient2
        self.global_orient3.data = global_orient3
        self.w.data = w
        self.angle.data = angle

        return loss


def backward_hook(self, grad_inputs, grad_outputs, scale=1):
    new_grad_inputs = tuple([g * scale if g is not None else g for g in grad_inputs])
    return new_grad_inputs


class ScaleGradient(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        # For some reason, defining it in the backward() function doesn't work
        self.register_backward_hook(partial(backward_hook, scale=scale))

    def forward(self, x):
        return x


def scale_grad(x, scale):
    return ScaleGradient(scale)(x)


# ---- Main optimization function.
def optimize_findz(
    cfg,
    z_init,
    transl_init,
    global_orient_init,
    w_init,
    a_init,
    gan_body,
    gan_rh,
    num_iterations=400,
    display=True,
    task='',
    extras={},
    model_name='flex'
):
    """
    Given object pose + trained VPoser prior + trained GrabNet prior,
    Perform optimization over (z-space + w-space + transl + global_orient + angle) to find those inputs that create a human with matching right-hand joint/vertices constraint and other losses, where:
        - z-space       (32,) latent space of VPoser
        - w-space       (16,) latent space of GrabNet
        - transl        (3,)  subject global translation passed to SMPL-X
        - global_orient (3,)  subject global orientation passed to SMPL-X
        - angle         (9,)  angle for object BPS sampling in latent space of GrabNet

    :param cfg                      OmegaConf dict of ['cuda_id', 'smplx_dir', 'mano_dir', 'mimenet_dataset_dir', num_neurons', 'latentD', 'best_mnet', 'best_pgprior', 'sbj_meshes_dir', 'obj_meshes_dir', 'obj_dir]
    :param z_init                   Tensor on device (b, 32)
    :param transl_init              Tensor on device (b, 3)
    :param global_orient_init       Tensor on device (b, 3) -- could be either (yaw/pitch/roll) or 3 axis-angle components.
    :param w_init                   Tensor on device (b, 16)
    :param a_init                   Tensor on device (b, 3)
    :param gan_body                 instance of VPoser model
    :param gan_rh                   list of right-hand grasping model [CoarseNet, RefineNet]
    :param extras                   dict of extra information needed for loss computation that is not optimized (this includes obstacles).
    :param model_name               string with model name. Either 'flex' or 'latent'

    :return dout                    dict of ['pose_body', 'transl', 'global_orient', 'z', 'rh_verts'] -- batched
    :return best_loss_dict          dict of batched losses - e.g., ['total', 'lowermost_loss', 'rh_match_loss', 'penet_loss', 'obstacle_loss_in', 'obstacle_loss_out', 'gaze_loss', 'rh_obstacle_loss_in', 'rh_obstacle_loss_out', 'wrist_loss']
    :return losses_stages           (dict) of ['total', 'rh_match', 'obstacle_in'] where each value is list of n Tensors of size (b, num_iterations) each - useful for plotting graph; n is number of stages
    """
    # Set device.
    device = 'cuda:'+str(cfg.cuda_id) if cfg.cuda_id != -1 else 'cpu'
    bs = cfg.batch_size

    # Main model.
    model_class = FLEX if model_name == 'flex' else Latent
    model = model_class(
        cfg=cfg,
        z_init=z_init,
        transl_init=transl_init,
        global_orient_init=global_orient_init,
        w_init=w_init,
        a_init=a_init,
        gan_body=gan_body,
        gan_rh=gan_rh,
        task=task,
        extras=extras
    )

    # Setup optimizer.
    params_mlp, other_params = [], []
    for name, p in model.named_parameters():
        if 'mlp.' in name:
            params_mlp.append(p)
        else:
            other_params.append(p)
    lr = cfg.latent_lr
    lr_mlp_divisor = cfg.lr_mlp_divisor
    optimizer = torch.optim.Adam([{'params': params_mlp, 'lr': lr/lr_mlp_divisor}, {'params': other_params}], lr=lr)

    # Print number of parameters.
    vars_mnet = [var[1] for var in model.named_parameters()]
    mnet_n_params = sum(p.numel() for p in vars_mnet if p.requires_grad)
    if display: print('Total Trainable Parameters for FLEX is %d.' % mnet_n_params)

    # Preliminaries.
    best_loss = torch.Tensor([np.inf for _ in range(bs)]).to(device)
    if display:
        loop = tqdm(range(num_iterations)) if display else range(num_iterations)
    else:
        loop = range(num_iterations) if display else range(num_iterations)

    # Initialization.
    best_z, best_w, best_angle = torch.zeros(bs,32).to(device), torch.zeros(bs,16).to(device), torch.zeros(bs,3).to(device)
    best_t1, best_t2, best_t3 = torch.zeros(bs).to(device), torch.zeros(bs).to(device), torch.zeros(bs).to(device)
    best_go1, best_go2, best_go3 = torch.zeros(bs).to(device), torch.zeros(bs).to(device), torch.zeros(bs).to(device)
    best_loss_dict = {'total': torch.zeros(bs).to(device),
                      'lowermost_loss': torch.zeros(bs).to(device),
                      'rh_match_loss': torch.zeros(bs).to(device),
                      'obstacle_loss_in': torch.zeros(bs).to(device),
                      'obstacle_loss_out': torch.zeros(bs).to(device),
                      'gaze_loss': torch.zeros(bs).to(device),
                      'wrist_loss': torch.zeros(bs).to(device),
                      'rh_obstacle_loss_in': torch.zeros(bs).to(device),
                      'rh_obstacle_loss_out': torch.zeros(bs).to(device)}
    # Stage specific.
    stages = cfg.num_stagewise.split(',')                                                               # e.g., ['3', '1']
    stage_params = cfg.params_stagewise.split(',')                                                      # e.g., ['tgz', 'w']
    stage_lrs = cfg.lrs_stagewise.split(',')                                                            # e.g., ['5e-2', '5e-4']
    # ---
    losses_stages_total = [torch.zeros(num_iterations, int(s), bs) for _,s in enumerate(stages)]        # e.g., [torch.zeros(n_iter, 3, bs), torch.zeros(n_iter, 1, bs)]
    losses_stages_rh_match = [torch.zeros(num_iterations, int(s), bs) for _,s in enumerate(stages)]     # e.g., [torch.zeros(n_iter, 3, bs), torch.zeros(n_iter, 1, bs)]
    losses_stages_obstacle_in = [torch.zeros(num_iterations, int(s), bs) for _,s in enumerate(stages)]  # e.g., [torch.zeros(n_iter, 3, bs), torch.zeros(n_iter, 1, bs)]
    # ---

    # Main loop.
    for idx in loop:
        if idx == 300:
            optimizer.param_groups[0]['lr'] = lr/lr_mlp_divisor/5
            optimizer.param_groups[1]['lr'] = lr/5
            cfg.alpha_wrist = 0.0  # It is only necessary to start finding the position

        # (*) Update per stage.
        loss, loss_batched = torch.tensor(0.0).to(device), torch.zeros(bs).to(device)
        for sid, stage_idx in enumerate(stages):

            # --> Set learning rate for this stage.
            if model_name is 'flex':
                params = stage_params[sid]
                for i, g in enumerate(optimizer.param_groups):
                    if i == 1: g['lr'] = float(stage_lrs[sid])
            else:
                params = cfg.latent_params

            # --> Set inner loop.
            stage_loop = tqdm(range(int(stage_idx))) if num_iterations == 1 else range(int(stage_idx))
            for sit in stage_loop:
                optimizer.zero_grad()
                loss_dict = model(mode=params,
                                    alpha_lowermost=cfg.alpha_lowermost,
                                    alpha_rh_match=cfg.alpha_rh_match,
                                    alpha_obstacle_in=cfg.alpha_obstacle_in,
                                    alpha_obstacle_out=cfg.alpha_obstacle_out,
                                    alpha_gaze=cfg.alpha_gaze,
                                    alpha_rh_obstacle_in=cfg.alpha_rh_obstacle_in,
                                    alpha_rh_obstacle_out=cfg.alpha_rh_obstacle_out,
                                    alpha_wrist=cfg.alpha_wrist,
                                    prediction_scale=cfg.prediction_scale,
                                    gradient_scale=cfg.gradient_scale,
                                    )
                curr_loss_batched = loss_dict['total']
                curr_loss = curr_loss_batched.mean()
                curr_loss.backward()
                optimizer.step()

                # Save for plotting.
                losses_stages_total[sid][idx][sit] = curr_loss_batched.cpu().detach()
                losses_stages_rh_match[sid][idx][sit] = loss_dict['rh_match_loss'].cpu().detach()
                losses_stages_obstacle_in[sid][idx][sit] = loss_dict['obstacle_loss_in'].cpu().detach()

            # --> Combine
            loss_batched += curr_loss_batched
            loss += curr_loss

            if cfg.save_every_step:
                # Save result at the end of each step
                current_trans = torch.stack([model.transl_x, model.transl_y, model.transl_z], 1)
                current_orient = recompose_angle(model.global_orient1, model.global_orient2, model.global_orient3, 'aa')
                dout = model.get_dout(current_trans, current_orient, model.z, model.w, model.angle, extras)
                dout = {k: dout[k].detach() for k in dout.keys()}
                curr_res = {'transl_init': transl_init, 'global_orient_init': global_orient_init,
                            'pose_final': dout['pose_body'], 'transl_final': dout['transl'],
                            'global_orient_final': dout['global_orient'],
                            'rh_verts': dout['rh_verts'], 'z_final': model.z.detach()}
                idx_res = {k: v.cpu().detach().numpy() for k, v in curr_res.items() if k != 'loss_dict'}
                idx_res['loss_dict'] = {k: v.cpu().detach().numpy() for k, v in loss_dict.items()}
                folder = f'save/{extras["obj_name"]}/'
                torch.save(idx_res, folder + f'{model_name}_iter{idx}.pth')

        # (*) Display combined loss of current run.
        if display:
            loop.set_description(f"Loss {loss.data:.4f}")
            if idx == 0: print("Starting TTO loss: %.2f" % loss.item())

        # (*) Save / make replacement for best model (z, w, transl, orient) for each batch element.
        replace = (loss_batched < best_loss).to(device)   # (bs)
        best_z = torch.where(replace[:, None], model.z, best_z)
        best_w = torch.where(replace[:, None], model.w, best_w)
        best_angle = torch.where(replace[:, None], model.angle, best_angle)
        best_go1, best_go2, best_go3 = torch.where(replace, model.global_orient1, best_go1), torch.where(replace, model.global_orient2, best_go2), torch.where(replace, model.global_orient3, best_go3)
        best_t1, best_t2, best_t3 = torch.where(replace, model.transl_x, best_t1), torch.where(replace, model.transl_y, best_t2), torch.where(replace, model.transl_z, best_t3)
        if cfg.orient_optim_type == 'aa':
            best_orient = torch.stack([best_go1, best_go2, best_go3], 1)
        else:  #'ypr'
            best_orient = recompose_angle(best_go1, best_go2, best_go3, 'aa')
        best_transl = torch.stack([best_t1, best_t2, best_t3], 1)
        best_loss = torch.where(replace, loss_batched, best_loss)
        for k, v in loss_dict.items():
            best_loss_dict[k] = torch.where(replace, v, best_loss_dict[k])

        if idx in cfg.iteration_filter:
            # Stop running unnecessary samples.
            half_good = loss_batched <= torch.quantile(loss_batched, 0.5)
            new_batch_size = half_good.sum().item()
            cfg['batch_size'] = new_batch_size
            model.bs = new_batch_size

            model.losses.rh_m.batch_size = new_batch_size
            model.losses.rh_m.betas = nn.Parameter(model.losses.rh_m.betas[half_good], requires_grad=True)
            model.losses.rh_m.transl = nn.Parameter(model.losses.rh_m.transl[half_good], requires_grad=True)
            model.losses.rh_m.global_orient = nn.Parameter(model.losses.rh_m.global_orient[half_good], requires_grad=True)

            model.losses.gan_rh_refine.rhm_train.batch_size = new_batch_size
            model.losses.gan_rh_refine.rhm_train.betas = nn.Parameter(model.losses.gan_rh_refine.rhm_train.betas[half_good], requires_grad=True)
            model.losses.gan_rh_refine.rhm_train.transl = nn.Parameter(model.losses.gan_rh_refine.rhm_train.transl[half_good], requires_grad=True)
            model.losses.gan_rh_refine.rhm_train.global_orient = nn.Parameter(model.losses.gan_rh_refine.rhm_train.global_orient[half_good], requires_grad=True)

            model.losses.sbj_m.batch_size = new_batch_size
            model.losses.sbj_m.jaw_pose = nn.Parameter(model.losses.sbj_m.jaw_pose[half_good], requires_grad=True)
            model.losses.sbj_m.leye_pose = nn.Parameter(model.losses.sbj_m.leye_pose[half_good], requires_grad=True)
            model.losses.sbj_m.reye_pose = nn.Parameter(model.losses.sbj_m.reye_pose[half_good], requires_grad=True)
            model.losses.sbj_m.expression = nn.Parameter(model.losses.sbj_m.expression[half_good], requires_grad=True)
            model.losses.sbj_m.left_hand_pose = nn.Parameter(model.losses.sbj_m.left_hand_pose[half_good], requires_grad=True)
            model.losses.sbj_m.right_hand_pose = nn.Parameter(model.losses.sbj_m.right_hand_pose[half_good], requires_grad=True)
            model.losses.sbj_m.betas = nn.Parameter(model.losses.sbj_m.betas[half_good], requires_grad=True)
            model.losses.sbj_m.global_orient = nn.Parameter(model.losses.sbj_m.global_orient[half_good], requires_grad=True)
            model.losses.sbj_m.body_pose = nn.Parameter(model.losses.sbj_m.body_pose[half_good], requires_grad=True)
            model.losses.sbj_m.transl = nn.Parameter(model.losses.sbj_m.transl[half_good], requires_grad=True)

            model.latent_z = nn.Parameter(model.latent_z[half_good].data, requires_grad=True)

            for sid in range(len(stages)):
                losses_stages_total[sid] = losses_stages_total[sid][:, :, half_good]
                losses_stages_rh_match[sid] = losses_stages_rh_match[sid][:, :, half_good]
                losses_stages_obstacle_in[sid] = losses_stages_obstacle_in[sid][:, :, half_good]

            bs = new_batch_size
            best_loss = best_loss[half_good]
            best_z = best_z[half_good]
            best_w = best_w[half_good]
            best_angle = best_angle[half_good]
            best_go1, best_go2, best_go3 = best_go1[half_good], best_go2[half_good], best_go3[half_good]
            best_t1, best_t2, best_t3 = best_t1[half_good], best_t2[half_good], best_t3[half_good]
            best_loss_dict = {k: v[half_good] for k, v in best_loss_dict.items()}

    if display:
        print("Ending TTO loss: %.5f" % loss_batched.mean().item())
        print("Best   TTO loss: %.5f" % best_loss.mean())


    # Decode to save final results.
    dout = model.get_dout(best_transl, best_orient, best_z, best_w, best_angle, extras)
    best_loss_dict = {k: v.cpu().detach() for k,v in best_loss_dict.items()}    # dict with keys ['total', 'lowermost_loss', 'rh_match_loss', 'obstacle_loss_in', 'obstacle_loss_out', 'gaze_loss'] -- each value of size (bs,

    # Save losses - each is a list of size number-of-iterations each a Tensor of size (bs, n_iter) -- take average across number of steps.
    losses_stages = {
        'total': [s.mean(1).T for s in losses_stages_total],
        'rh_match': [s.mean(1).T for s in losses_stages_rh_match],
        'obstacle_in': [s.mean(1).T for s in losses_stages_obstacle_in]
    }

    return dout, best_loss_dict, losses_stages