# Underwater Image Enhancement Baselines

This repository provides implementation of some underwater image enhancement methods and datasets, including:

**Supported Methods**
| Method | Pub | Language | Paper | Reference Codes |
| ------ | ------- | ----- | ----- | --------------- |
| Fusion | 2012 CVPR | MATLAB | [Enhancing Underwater Images and Videos by Fusion](https://ieeexplore.ieee.org/abstract/document/6247661) | [Fusion](https://github.com/bilityniu/underwater_image_fusion) |
| UWCNN | 2020 PR | Pytorch | [Underwater Scene Prior Inspired Deep Underwater Image and Video Enhancement](https://www.sciencedirect.com/science/article/abs/pii/S0031320319303401) | [UWCNN](https://github.com/BIGWangYuDong/UWEnhancement/blob/master/core/Models/UWModels/UWCNN.py) |
| UIEC2Net | 2021 SPIC | Pytorch | [UIEC^2-Net: CNN-based Underwater Image Enhancement Using Two Color Space](https://arxiv.org/abs/2103.07138) | [UIEC2Net](https://github.com/BIGWangYuDong/UWEnhancement/blob/master/core/Models/UWModels/UIEC2Net.py) |
| MLLE | 2022 TIP | MATLAB | [Underwater Image Enhancement via Minimal Color Loss and Locally Adaptive Contrast Enhancement](https://ieeexplore.ieee.org/abstract/document/9788535) | [MLLE](https://github.com/Li-Chongyi/MMLE_code) |
| UTrans | 2023 TIP | Pytorch | [U-Shape Transformer for Underwater Image Enhancement](https://arxiv.org/abs/2111.11843) | [UTrans](https://github.com/LintaoPeng/U-shape_Transformer_for_Underwater_Image_Enhancement) |
| NU2Net | 2023 AAAI | Pytorch | [Underwater Ranker: Learn Which Is Better and How to Be Better](https://arxiv.org/abs/2208.06857) | [NU2Net](https://github.com/RQ-Wu/UnderwaterRanker) |
| FiveAPlus | 2023 BMVC | Pytorch | [Five A+ Network: You Only Need 9K Parameters for Underwater Image Enhancement](https://arxiv.org/abs/2305.08824) | [FiveAPlus](https://github.com/Owen718/FiveAPlus-Network) |


**Supported Datasets**
| Dataset | Link |
| ------ | ------- |
| UIEB | https://li-chongyi.github.io/proj_benchmark.html | 
| U45 | https://github.com/IPNUISTlegal/underwater-test-dataset-U45- | 
| LSUI | https://github.com/LintaoPeng/U-shape_Transformer_for_Underwater_Image_Enhancement | 


## Environment

```
conda create -n uie python=3.9
conda activate uie

pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install pyiqa

pip install pytorch_lightning==2.0.9.post0
```

## Prepare Data
Download the UIEB dataset in the `./data/UIEB/` folder, then you have three subfolders `raw-890/`, `reference-890/`, `challenging-60/`, just like:
```
./data/UIEB/
├── challenging-60/
├── challenging.txt
├── raw-890/
├── reference-890/
├── test.txt
└── train.txt
```
Download the LSUI dataset in the `./data/LSUI/` folder, then you have two subfolders `GT/`, `input/`, just like:
```
./data/LSUI/
├── GT/
├── input/
├── test.txt
└── train.txt
```

## Train
Train on the training set of UIEB dataset:
```
python train_UIEB.py --model_name UIEC2Net --batch_size 16 --epochs 100
```
We provide four models' checkpoints trained on UIEB dataset in the `./checkpoints/UIEB/` folder:
```
./checkpoints/UIEB/
├── FIVE_APLUS.ckpt
├── NU2Net.ckpt
├── UIEC2Net.ckpt
└── UWCNN.ckpt
```
## Test
After training, you can enhance the images in the test set and challenging set of UIEB dataset:
```
python test_UIEB.py --model_name UIEC2Net
```
The generated enhanced images are saved in the `./data/UIEB/All_Results/` folder.
The folder's structure is like:
```
./data/UIEB/All_Results/
├── FIVE_APLUSNet
│   ├── C60/
│   └── T90/
├── NU2Net
│   ├── C60/
│   └── T90/
├── UIEC2Net
│   ├── C60/
│   └── T90/
├── UTrans
│   ├── C60/
│   └── T90/
└── UWCNN
    ├── C60/
    └── T90/
```
Each subfolder corresponds to the results of one method.

## Evaluation
After testing, you can evaluate any method's perfermance on the test set of UIEB dataset:
```
python evaluate_UIEB.py --method_name UIEC2Net --folder T90
```
or you can evaluate perfermance on the challenging set of UIEB dataset:
```
python evaluate_UIEB.py --method_name UIEC2Net --folder C60
```
## Citation

```
@article{du2023uiedp,
  title={UIEDP: Underwater Image Enhancement with Diffusion Prior},
  author={Du, Dazhao and Li, Enhan and Si, Lingyu and Xu, Fanjiang and Niu, Jianwei and Sun, Fuchun},
  journal={arXiv preprint arXiv:2312.06240},
  year={2023}
}
```
