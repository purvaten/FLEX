This repo is the official implementation of the paper "FLEX: Full-Body Grasping Without Full-Body Grasps".


<p align="center">

  <h1 align="center">FLEX: Full-Body Grasping Without Full-Body Grasps</h1>
  <p align="center">
    <a href="https://purvaten.github.io/"><strong>Purva Tendulkar</strong></a>
    ·
    <a href="https://www.didacsuris.com/"><strong>Dídac Surís</strong></a>
    ·
    <a href="http://www.cs.columbia.edu/~vondrick/"><strong>Carl Vondrick</strong></a>
  </p>

  <a href="">
    <img src="./images/teaser.png" alt="Logo" width="100%">
  </a>


<p align="center">
  <br>
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
    <a href='https://arxiv.org/abs/2211.11903'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='Paper PDF'>
    </a>
    <a href='https://flex.cs.columbia.edu/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'><br></br>

[//]: # (<a href='https://arxiv.org/abs/2211.11903'>)

[//]: # (      <img src='https://img.shields.io/badge/Paper-PDF-green?style=for-the-badge&logo=arXiv&logoColor=green' alt='Paper PDF'>)

[//]: # (    </a>)

[//]: # (    <a href='https://icon.is.tue.mpg.de/' style='padding-left: 0.5rem;'>)

[//]: # (      <img src='https://img.shields.io/badge/ICON-Page-orange?style=for-the-badge&logo=Google%20chrome&logoColor=orange' alt='Project Page'>)

[//]: # (    <a href="https://discord.gg/Vqa7KBGRyk"><img src="https://img.shields.io/discord/940240966844035082?color=7289DA&labelColor=4a64bd&logo=discord&logoColor=white&style=for-the-badge"></a>)

[//]: # (    <a href="https://youtu.be/hZd6AYin2DE"><img alt="youtube views" title="Subscribe to my YouTube channel" src="https://img.shields.io/youtube/views/ZufrPvooR2Q?logo=youtube&labelColor=ce4630&style=for-the-badge"/></a>)
  </br>
    
   
  </p>
</p>



[//]: # (### Run in Google-Colab)

[//]: # ([![Open In Google-Colab]&#40;https://colab.research.google.com/assets/colab-badge.svg&#41;]&#40;https://colab.research.google.com/drive/1c8OfrHhkwtyb0m75QHO0Dpj1IAgDcaqQ&#41;)


[FLEX](http://flex.cs.columbia.edu) is a generative model that generates full-body avatars grasping 3D objects in a 3D environment. FLEX leverages the existence of pre-trained prior models for:
1. **Full-Body Pose** - [VPoser](https://github.com/nghorbani/human_body_prior) (trained on the [AMASS](https://amass.is.tue.mpg.de/index.html) dataset)
2. **Right-Hand Grasping** - [GrabNet](https://github.com/otaheri/GrabNet) (trained on right-handed grasps of the [GRAB](http://grab.is.tue.mpg.de) dataset)
3. **Pose-Ground Relation** - [PGPrior](https://github.com/purvaten/FLEX) (trained on the [AMASS](https://amass.is.tue.mpg.de/index.html) dataset)

For more details please refer to the [Paper](https://arxiv.org/abs/2211.11903) or the [project website](http://flex.cs.columbia.edu).



<!-- For more details check out the YouTube video below.
[![Video](images/video_teaser_play.png)](https://www.youtube.com/watch?v=A7b8DYovDZY) -->



## Table of Contents
  * [Description](#description)
  * [Requirements](#requirements)
  * [Installation](#installation)
  * [Getting Started](#getting-started)
  * [Examples](#examples)
  * [Citation](#citation)
  * [Acknowledgments](#acknowledgments)
  * [Contact](#contact)



## Description

This implementation:

- Can run FLEX on arbitrary objects in arbitrary scenes provided by users.
- Can run FLEX on the test objects of the GRAB dataset (with pre-computed object centering and BPS representation).


## Requirements
This package has been tested for the following:

* [Pytorch>=1.7.1](https://pytorch.org/get-started/locally/) 
* Python >=3.7.0
* [MANO](https://github.com/otaheri/MANO) 
* [SMPLX](https://github.com/vchoutas/smplx) 
* [chamfer_distance](https://github.com/otaheri/chamfer_distance)
* [bps_torch](https://github.com/otaheri/bps_torch)
* [psbody-mesh](https://github.com/MPI-IS/mesh)
* [Kaolin](https://github.com/NVIDIAGameWorks/kaolin)

## Installation

To install the dependencies please follow the next steps:

- Clone this repository: 
    ```Shell
    git clone https://github.com/purvaten/FLEX.git
    cd FLEX
    ```
- Install the dependencies by the following commands:
    ```
    conda create -n flex python=3.7.11
    conda activate flex
    conda install pytorch==1.10.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch
    conda install pytorch3d -c pytorch3d
    conda install meshplot
    conda install -c conda-forge jupyterlab
    pip install -r requirements.txt
    pip install kaolin==0.12.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.10.1_cu113.html
    ```

## Getting started

In order to run FLEX, create a `data/` directory and follow the steps below:

#### ReplicaGrasp Dataset
- Download the Habitat receptacle meshes info from [here](https://flex.cs.columbia.edu/data/receptacles.npz). This is dictionary where the keys are the names of the receptacles, and the values are a list of `[vertices, faces]` for different configurations of that receptacle (e.g., with doors open, doors closed, drawer opened, etc.)
- Download the main dataset from [here](https://flex.cs.columbia.edu/data/dset_info.npz). This is a dictionary where the keys are the names of the example instances, and the values are the `[object_translation, object_orientation, recept_idx]` where `recept_idx` is the index to the receptacle configuration in `receptacles.npz`.
- Store both files under `FLEX/data/replicagrasp/`.
- To visualize random instances of the dataset, run the notebook `FLEX/flex/notebooks/viz_replicagrasp.ipynb`.


#### Dependecy Files
- Download the SMPL-X and MANO models from the [SMPL-X website](https://smpl-x.is.tue.mpg.de/) and [MANO website](https://mano.is.tue.mpg.de/).
- Download the GRAB object mesh (`.ply`) files and BPS points (`bps.npz`) from the [GRAB website](https://grab.is.tue.mpg.de/). Download `obj_info.npy` from [here](https://flex.cs.columbia.edu/data/obj_info.npy).
- Download full-body related desiderata [here](https://flex.cs.columbia.edu/data/sbj.zip).
- The final structure of data should look as below:
```bash
    FLEX
    ├── data
    │   │
    │   ├── smplx_models
    │   │       ├── mano
    │   │       │     ├── MANO_LEFT.pkl
    │   │       │     ├── MANO_RIGHT.pkl
    │   │       └── smplx
    │   │             ├── SMPLX_FEMALE.npz
    │   │             └── ...
    │   ├── obj
    │   │    ├── obj_info.npy
    │   │    ├── bps.npz
    │   │    └── contact_meshes
    │   │             ├── airplane.ply
    │   │             └── ...
    │   ├── sbj
    │   │    ├── adj_matrix_original.npy
    │   │    ├── adj_matrix_simplified.npy
    │   │    ├── faces_simplified.npy
    │   │    ├── interesting.npz
    │   │    ├── MANO_SMPLX_vertex_ids.npy
    │   │    ├── sbj_verts_region_mapping.npy
    │   │    └── vertices_simplified_correspondences.npy
    │   │
    │   └── replicagrasp
    │        ├── dset_info.npz
    │        └── receptacles.npz
    .
    .
```

#### Pre-trained Checkpoints
- Download the VPoser prior (`VPoser v2.0`) from the [SMPL-X website](https://smpl-x.is.tue.mpg.de/).
- Download the checkpoints of the hand-grasping pre-trained model (`coarsenet.pt` and `refinenet.pt`) from the [GRAB website](https://grab.is.tue.mpg.de/).
- Download the pose-ground prior from [here](https://flex.cs.columbia.edu/data/pgp.pth).
- Place all pre-trained models in `FLEX/flex/pretrained_models/ckpts` as follows:
```bash
    ckpts
    ├── vposer_amass
    │   │
    │   ├── snapshots
    │   │       └── V02_05_epoch=13_val_loss=0.03
    │   ├── V02_05.log
    │   └── V02_05.yaml
    │
    ├── vposer_grab.pt
    ├── coarsenet.pt
    ├── refinenet.pt
    └── pgp.pth
```


## Examples

After installing the *FLEX* package, dependencies, and downloading the data and the models, you should be able to run the following examples:
                                            

- #### Generate whole-body grasps for ReplicaGrasp.
  
    ```Shell
    python run.py \
    --obj_name stapler \
    --receptacle_name receptacle_aabb_TvStnd1_Top3_frl_apartment_tvstand \
    --ornt_name all \
    --gender 'female'
    ```
    The result will be saved in `FLEX/save`. The optimization for an example should take 7-8 minutes on a single RTX Ti 2080.
    
- #### Visualize the result by running the jupyter notebook `FLEX/flex/notebooks/viz_results.ipynb`.


## Citation

```
@inproceedings{tendulkar2022flex,
    title = {FLEX: Full-Body Grasping Without Full-Body Grasps},
    author = {Tendulkar, Purva and Sur\'is, D\'idac and Vondrick, Carl},
    booktitle = {Conference on Computer Vision and Pattern Recognition ({CVPR})},
    year = {2023},
    url = {https://flex.cs.columbia.edu/}
}
```


## Acknowledgments

This research is based on work partially supported by NSF NRI Award #2132519, and the DARPA MCS program under Federal Agreement No. N660011924032. Dídac Surís is supported by the Microsoft PhD fellowship. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of the sponsors.

We thank: [Alexander Clegg](https://www.linkedin.com/in/alexander-clegg-68336839/) for helping with Habitat-related questions and [Harsh Agrawal](https://dexter1691.github.io/) for helpful discussions and feedback.

This template was adapted from the GitHub repository of [GOAL](https://github.com/otaheri/GOAL).

## Contact
The code of this repository was implemented by [Purva Tendulkar](https://purvaten.github.io/) and [Dídac Surís](https://www.didacsuris.com/).

For questions, please contact [pt2578@columbia.edu](mailto:pt2578@columbia.edu).