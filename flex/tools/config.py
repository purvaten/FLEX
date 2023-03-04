from dataclasses import dataclass, field
from typing import *

@dataclass
class Config:
    # --- GENERAL --- #
    best_mnet_amass: str = ''                                                               # Path to pretrained VPoser prior (on AMASS)
    best_mnet_grab: str = ''                                                                # Path to pretrained VPoser prior (on GRAB)
    cuda_id: int = 0
    notebook: bool = False                                                                  # whether code is run from notebook or not
    save_every_step: bool = False                                                           # whether to save every step or not
    model_name: str = 'flex'
    gender: str = 'neutral'
    bs: int = 3125
    # --- PATH: Datasets & Meshes --- #
    smplx_dir: str = 'data/smplx_models'
    mano_dir: str = 'data/smplx_models/mano'
    obj_meshes_dir: str = 'data/obj/contact_meshes/'
    obj_dir: str = 'data/obj/obj_info.npy'
    bps_pth: str = 'data/obj/bps.npz'
    sbj_verts_region_map_pth: str = 'data/sbj/sbj_verts_region_mapping.npy'
    mano2smplx_verts_ids: str = 'data/sbj/MANO_SMPLX_vertex_ids.npy'
    interesting_pth: str = 'data/sbj/interesting.npz'
    sbj_verts_simplified: str = 'data/sbj/vertices_simplified_correspondences.npy'
    sbj_faces_simplified: str = 'data/sbj/faces_simplified.npy'
    adj_matrix_orig: str = 'data/sbj/adj_matrix_original.npy'
    adj_matrix_simplified: str = 'data/sbj/adj_matrix_simplified.npy'
    # --- VPOSER --- #
    num_neurons: int = 512
    latentD: int = 32
    vposer_dset: str = 'grab'                                                               # Whether loaded VPoser prior should be pre-trained on AMASS or GRAB only
    # --- TTO --- #
    obj_name: str = 'hammer'                                                                # object to grasp
    topk: int = 5                                                                           # how many top results to return
    std: int = -1                                                                           # -1 means random sample, anything else will sample those many SDs away from mean
    save_all_meshes: bool = False                                                           # Save all meshes of run.py (should be set to True for debugging only)
    alpha_rh_match: float = 3.0                                                             # factor to weigh right-hand matching loss term
    rh_match_type: str = 'joints'                                                           # joints / verts -- determines matching loss criteria
    alpha_interesting_verts: float = 0.0                                                    # factor to weigh interesting vertices loss term
    alpha_lowermost: float = 0.2                                                            # factor to weigh lowermost loss term
    alpha_obstacle_in: float = 100.0                                                        # factor to weigh human-obstacle penetration loss term
    alpha_obstacle_out: float = 0.0                                                         # factor to weigh human-obstacle faraway loss term for the (opposite of penetration)
    alpha_rh_obstacle_in: float = 0.0
    alpha_rh_obstacle_out: float = 0.0
    alpha_wrist: float = 0.0
    intersection_thresh: float = 0.0                                                        # threshold for new intersection loss
    obstacle_sbj2obj_extra: str = ''                                                        # 'connected_components' / '' (i.e.,none) -- only valid for human, not rhand
    obstacle_sbj2obj: bool = True                                                           # whether to have obstacle loss from subject-to-object
    obstacle_obj2sbj: bool = False                                                          # whether to have obstacle loss from object-to-subject
    alpha_gaze: float = 1.0                                                                 # factor to weigh human-object gaze loss term
    disp_t: bool = True                                                                     # Change translation of object during displacement to test-pose
    disp_g: bool = True                                                                     # Change global orientation of object during displacement to test-pose
    orient_optim_type: str = 'ypr'                                                          # How to optimize for global orientationg - either 'ypr' for 'yaw-pitch-roll' or 'aa' for axis-angle
    subsample_sbj: bool = True                                                              # Subsample subject vertices for penetration computation speed-up
    n_iter: int = 50                                                                        # Number of iterations of optimization
    # --- POSE-GROUND PRIOR --- #
    pgprior: bool = True                                                                    # Whether to use pose-prior loss or not (turn off for baseline)
    best_pgprior: str = ''                                                                  # Path to pretrained pose-ground prior
    height: str = 'transl_z'                                                                # either of 'transl_z' or 'pelvis_z' - depending on what pretrained model is loaded
    # --- STAGEWISE TRAINING --- #
    num_stagewise: str = '1'                                                                # Number of optimization iters for different stages
    params_stagewise: str = 'tgzw'
    lrs_stagewise: str = '5e-2'                                                             # Learning rate for different stages of optimization
    # --- LATENT SPACE TRAINING --- #
    latent_lr: float = 5e-2                                                                 # Common learning rate for latent-based optimization (Roshi's paper)
    lr_mlp_divisor: float = 10.0                                                            # Learning rate for MLP is divided by this factor
    latent_params: str = 'tgzwa'
    prediction_scale: Dict[str, str] = field(default_factory=lambda:
        {'z': 5, 'transl': 10, 'orient': 20, 'w': 1, 'angle': 20})                          # Scale for different parameters in latent space
    gradient_scale: Dict[str, str] = field(default_factory=lambda:
        {'z': 10, 'transl': 0.3, 'orient': 1, 'w': 1, 'angle': 1})                          # Scale for different gradients in latent space
    iteration_filter: List[int] = field(default_factory=lambda: [100, 200, 300])
