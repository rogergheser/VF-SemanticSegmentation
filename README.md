# Trends and Applications of Computer vision [2024]
This is the repo for the course Trends and Applications of Computer Vision at the University of Trento. The course is part of the Master in Artificial Intelligence.

# Installation
## Requirements
To run the experiments you need python 3.10

## Install dependencies
```bash
sh setup.sh
```
## Download dataset
First configure appropiately the 'datasets.yaml' file. Download the missing values and then run the following commands:
```bash
python download_dataset.py
# After manually getting the missing values
sh preprocess_dataset.sh
```
## Download models
Note:
AlphaCLIP only has google drive link working, so you need to download it manually and place it in the 'models' folder.

* SAN           -- [model zoo](https://github.com/MendelXu/SAN?tab=readme-ov-file#pretrained-weights)   -- [default](https://huggingface.co/Mendel192/san/resolve/main/san_vit_b_16.pth)
* SAM           -- [model zoo](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)   -- [default](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
* AlphaCLIP     -- [model zoo](https://github.com/SunzeY/AlphaCLIP/blob/main/model-zoo.md)  -- [default](https://drive.google.com/file/d/11iDlSAYI_BAi1A_Qz6LTWYHNgPe-UY7I/view?usp=sharing)



## Vocabulary Free Semantic Segmentation

### SOTA

### SAM

### SAN

### SAM vs SAN