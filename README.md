# [CVPR 2025] Progressive Focused Transformer for Single Image Super-Resolution

This repository is an official implementation of the paper "Progressive Focused Transformer for Single Image Super-Resolution", CVPR, 2025.

[[arXiv](https://arxiv.org/abs/2503.20337)] [[Pretrained Models](https://drive.google.com/drive/folders/1ChkxVDghFWUtJydJKLp5yssrUfm0VWfg?usp=sharing)]

By [Wei Long](https://scholar.google.com/citations?user=CsVTBJoAAAAJ), [Xingyu Zhou](https://scholar.google.com/citations?user=dgO3CyMAAAAJ), [Leheng Zhang](https://scholar.google.com/citations?user=DH1CJqkAAAAJ), and [Shuhang Gu](https://scholar.google.com/citations?user=-kSTt40AAAAJ).

> **Abstract:** Transformer-based methods have achieved remarkable results in image super-resolution tasks because they can capture non-local dependencies in low-quality input images. However, this feature-intensive modeling approach is computationally expensive because it calculates the similarities between numerous features that are irrelevant to the query features when obtaining attention weights. These unnecessary similarity calculations not only degrade the reconstruction performance but also introduce significant computational overhead. How to accurately identify the features that are important to the current query features and avoid similarity calculations between irrelevant features remains an urgent problem. To address this issue, we propose a novel and effective **P**rogressive **F**ocused **T**ransformer (**PFT**) that links all isolated attention maps in the network through Progressive Focused Attention (PFA) to focus attention on the most important tokens. PFA not only enables the network to capture more critical similar features, but also significantly reduces the computational cost of the overall network by filtering out irrelevant features before calculating similarities. Extensive experiments demonstrate the effectiveness of the proposed method, achieving state-of-the-art performance on various single image super-resolution benchmarks..
> 
> <img width="800" src="figures/pft_m.png"> 
> <br/>
> <img width="800" src="figures/PFT.png"> 



## Contents
1. [Enviroment](#environment)
1. [Inference](#inference)
1. [Training](#training)
1. [Testing](#testing)
1. [Results](#results)
1. [Visual Results](#visual-results)
1. [Acknowledgements](#acknowledgements)


## Environment
- Python 3.9
- PyTorch 2.5.1

### Installation
```bash
git clone https://github.com/LabShuHangGU/PFT-SR.git

conda create -n PFT python=3.9
conda activate PFT

pip install -r requirements.txt
python setup.py develop

cd ./ops_smm
./make.sh
```

## Inference
Using ```inference.py``` for fast inference on single image or multiple images within the same folder.
```bash
# For classical SR
python inference.py -i test_image.png -o results/test/ --scale 4 --task classical
python inference.py -i test_images/ -o results/test/ --scale 4 --task classical

# For lightweight SR
python inference.py -i test_image.png -o results/test/ --scale 4 --task lightweight
python inference.py -i test_images/ -o results/test/ --scale 4 --task lightweight
```
The PFT SR model processes the image ```test_image.png``` or images within the ```test_images/``` directory. The results will be saved in the ```results/test/``` directory.


## Training
### Data Preparation
- Download the training dataset DF2K ([DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) + [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)) and put them in the folder `./datasets`.
- It's recommanded to refer to the data preparation from [BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md) for faster data reading speed.

### Training Commands
- Refer to the training configuration files in `./options/train` folder for detailed settings.
- PFT (Classical Image Super-Resolution)
```bash
# batch size = 8 (GPUs) × 4 (per GPU)
# training dataset: DF2K

# ×2 scratch, input size = 64×64, 500k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --use-env --nproc_per_node=8 --master_port=1145  basicsr/train.py -opt options/train/001_PFT_SRx2_scratch.yml --launcher pytorch

# ×3 finetune, input size = 64×64, 250k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --use-env --nproc_per_node=8 --master_port=1145  basicsr/train.py -opt options/train/002_PFT_SRx3_finetune.yml --launcher pytorch

# ×4 finetune, input size = 64×64, 250k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --use-env --nproc_per_node=8 --master_port=1145  basicsr/train.py -opt options/train/003_PFT_SRx4_finetune.yml --launcher pytorch
```

- PFT-light (Lightweight Image Super-Resolution)
```bash
# batch size = 2 (GPUs) × 16 (per GPU)
# training dataset: DIV2K

# ×2 scratch, input size = 64×64, 500k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use-env --nproc_per_node=4 --master_port=1145  basicsr/train.py -opt options/train/101_PFT_light_SRx2_scratch.yml --launcher pytorch

# ×3 finetune, input size = 64×64, 250k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use-env --nproc_per_node=4 --master_port=1145  basicsr/train.py -opt options/train/102_PFT_light_SRx3_finetune.yml --launcher pytorch

# ×4 finetune, input size = 64×64, 250k iterations
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use-env --nproc_per_node=4 --master_port=1145  basicsr/train.py -opt options/train/103_PFT_light_SRx4_finetune.yml --launcher pytorch
```


## Testing
### Data Preparation
- Download the testing data (Set5 + Set14 + BSD100 + Urban100 + Manga109 [[download](https://drive.google.com/file/d/1_4Fy9emAcqdiBwVM6FvbJU50LCtaBoMt/view?usp=sharing)]) and put them in the folder `./datasets`.

### Pretrained Models
- Download the [pretrained models](https://drive.google.com/drive/folders/1ChkxVDghFWUtJydJKLp5yssrUfm0VWfg?usp=sharing) and put them in the folder `./experiments/pretrained_models`.

### Testing Commands
- Refer to the testing configuration files in `./options/test` folder for detailed settings.
- PFT (Classical Image Super-Resolution)
```bash
python basicsr/test.py -opt options/test/001_PFT_SRx2_scratch.yml
python basicsr/test.py -opt options/test/002_PFT_SRx3_finetune.yml
python basicsr/test.py -opt options/test/003_PFT_SRx4_finetune.yml
```

- PFT-light (Lightweight Image Super-Resolution)
```bash
python basicsr/test.py -opt options/test/101_PFT_light_SRx2_scratch.yml
python basicsr/test.py -opt options/test/102_PFT_light_SRx3_finetune.yml
python basicsr/test.py -opt options/test/103_PFT_light_SRx4_finetune.yml
```


## Results
- Classical Image Super-Resolution

<img width="800" src="figures/classical.png">

- Lightweight Image Super-Resolution

<img width="800" src="figures/lightweight.png">

## Visual Results

<img width="800" src="figures/visual_classical.png">

<img width="800" src="figures/visual_lightweight.png">


## Acknowledgements
This code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR) and [ATD](https://github.com/LabShuHangGU/Adaptive-Token-Dictionary.git).

