# MoCapDeform
Code and data release for the paper *MoCapDeform: Monocular 3D Human Motion Capture in Deformable Scenes*

## Prerequisites
Our code is tested on python==3.8.2 with pytorch==1.7.1, open3d==0.11.2 and trimesh==3.9.20

Other fundamental packages: numpy, scipy, opencv, matplotlib, sklearn, tqdm, json and pickle

Task specific packages: [smplx](https://github.com/vchoutas/smplx),
[detectron2](https://github.com/facebookresearch/detectron2),
[pyembree](https://github.com/scopatz/pyembree),
and [ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch)

## MoCapDeform Dataset
Details can be found [here](https://github.com/Malefikus/MoCapDeform/blob/main/dataset/README.md).
Please download the files as instructed for running the experiments.

## Running MoCapDeform
### Stage1: Initial Pose Estimation
We utilise [smplify-x](https://github.com/vchoutas/smplify-x) for initialisation.
The results are stored at dataset/subject/RGB.
To generate the smplify-x results, we need to get [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
2d keypoint detection results which are stored at dataset/subject/keypoints.
For [PROX](https://github.com/mohamedhassanmus/prox)
and [POSA](https://github.com/mohamedhassanmus/POSA) please refer to their code.

### Stage2: Global Pose Optimisation
#### Raycast
First generate body-centric human contacts with [POSA](https://github.com/mohamedhassanmus/POSA):
`python gen_contacts.py --config posa_contacts/contact.yaml`
