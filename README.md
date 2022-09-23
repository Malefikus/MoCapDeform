# MoCapDeform
Code and data release for the paper
[*MoCapDeform: Monocular 3D Human Motion Capture in Deformable Scenes*] (https://4dqv.mpi-inf.mpg.de/MoCapDeform/)
(3DV 2022 best student paper award :3).

## Prerequisites
Our code is tested on python==3.8.2 with pytorch==1.7.1, open3d==0.11.2 and trimesh==3.9.20

Other fundamental packages: numpy, scipy, opencv, matplotlib, sklearn, tqdm, json and pickle

Task specific packages: [smplx](https://github.com/vchoutas/smplx),
[detectron2](https://github.com/facebookresearch/detectron2),
[pyembree](https://github.com/scopatz/pyembree) (optional, to speed up raycasting by 50x),
and [ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch)

## MoCapDeform Dataset
Details can be found [here](https://github.com/Malefikus/MoCapDeform/blob/main/dataset/README.md).
Please download the files as instructed for running the experiments.

## Running MoCapDeform
### Download Models
MoCapDeform utilises several existing methods such as [smplify-x](https://github.com/vchoutas/smplify-x),
[PROX](https://github.com/mohamedhassanmus/prox),
[POSA](https://github.com/mohamedhassanmus/POSA) and
[PointRend](https://github.com/facebookresearch/detectron2).

We provide all necessary models in our `models` folder;
please download as instructed [here](https://github.com/Malefikus/MoCapDeform/blob/main/models/README.md).

All the models are provided by the original repositories.

### Stage1: Initial Pose Estimation
We utilise [smplify-x](https://github.com/vchoutas/smplify-x) for initialisation.
The results are stored at `dataset/subject/RGB`.

As the smplify-x optimisation doesn't always converge,
we store the converged frame number in `dataset/subject/avail_frms.npy`

To generate the smplify-x results, we need to get [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
2d keypoint detection results which are stored at `dataset/subject/keypoints`.

For [PROX](https://github.com/mohamedhassanmus/prox)
and [POSA](https://github.com/mohamedhassanmus/POSA) please refer to their code.

### Stage2: Global Pose Optimisation
First generate body-centric human contacts with [POSA](https://github.com/mohamedhassanmus/POSA):

```Shell
python gen_contacts.py --config posa_contacts/contact.yaml
```

Then generate tight human masks with [PointRend](https://github.com/facebookresearch/detectron2):

```Shell
python gen_mask.py
```

Next, get scene contacts by raycasting:

```Shell
python gen_raycast.py
```

At last, run stage2 optimisation:

```Shell
python optimise-stage2.py
```

Optimised human poses are then stored at `dataset/subject/stage2.npy`.

### Stage3: Joint Scene Deformation and Pose Refinement
Simply run `python optimise-stage3.py` for the optimisation.
The optimised human pose and scene deformation are stored at `dataset/subject/stage3.npy`

Note that MoCapDeform dataset assumes that all the furniture in the dataset are deformable.
For running on datasets such as [PROX](https://prox.is.tue.mpg.de/) or your own dataset,
you may want to do 3d semantic segmentation first to determine rigidity flags;
in the experiments in our paper we adopt the trained [VMNet](https://github.com/hzykent/VMNet) for segmentation.

## Citation
If you find our work useful, please kindly cite:

```
@inproceedings{Li_3DV2022,
    title={MoCapDeform: Monocular 3D Human Motion Capture in Deformable Scenes},
    author={Zhi Li and Soshi Shimada and Bernt Schiele and Christian Theobalt and Vladislav Golyanik},
    booktitle = {International Conference on 3D Vision (3DV)},
    year={2022}
}
```

## Contact
For questions, clarifications, please get in touch with:

Zhi Li
[zhili@mpi-inf.mpg.de](zhili@mpi-inf.mpg.de)

Soshi Shimada
[sshimada@mpi-inf.mpg.de](sshimada@mpi-inf.mpg.de)

Vladislav Golyanik
[golyanik@mpi-inf.mpg.de](golyanik@mpi-inf.mpg.de)
