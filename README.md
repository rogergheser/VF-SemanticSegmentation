<div align="center">
  <img src="https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white" alt="OpenCV"/>
</div>

<p align='center'>
    <h1 align="center">Open Vocabulary Semantic Segmentation</h1>
    <p align="center">
    Project for Trends and application of computer vision at the University of Trento A.Y.2024/2025
    </p>
    <p align='center'>
    Developed by: 
    Gheser Amir, 
    Mascherin Matteo,
    Roman Simone
    </p>   
</p>

----------

- [Project Description](#project-description)
  - [SAN](#san)
  - [SAM](#sam)
  - [Experiments](#experiments)
- [Installation](#installation)
- [Running the project](#running-the-project)

## Project Description
For this project, our goal is to explore advanced methods for open-vocabulary semantic segmentation (OVSS), aiming to segment images into regions defined by arbitrary textual concepts. Our research investigates two main approaches: SAN (Side Network) and SAM (Segment Anything), comparing their performance and adaptability in OVSS tasks.

### SAN
The <a href="https://arxiv.org/abs/2302.12242">Side Adapter Network (SAN)</a> is a lightweight framework designed for open-vocabulary semantic segmentation, leveraging CLIP's pre-trained vision-language capabilities. SAN models segmentation as a region recognition task by attaching a side network to CLIP with two branches: one for mask proposals and the other for attention bias, enabling CLIP-aware segmentation. Its end-to-end training maximizes adaptation to CLIP, ensuring accurate, efficient predictions. Compared to alternatives, SAN achieves state-of-the-art performance with up to 18x fewer parameters and 19x faster inference. It excels in resource efficiency while delivering high-quality segmentation across diverse datasets.


### SAM

<a href = "https://segment-anything.com/">Segment anything (SAM)</a> is the state of the art AI framework for object segmentation across diverse domains. To adapt SAM to OVSS task we propose a two-stage approach where SAM acts as a class-agnostic mask generator, and <a href = "https://arxiv.org/abs/2312.03818">Alpha-CLIP</a> is employed for mask classification. Post-processing techniques, such as BBox filtering and background adjustments, refine the mask proposals for enhanced segmentation accuracy in open-vocabulary settings.

### Experiments



## Installation

## Running the project