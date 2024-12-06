import torch
import torchvision
import torch.utils.data as data
import matplotlib.pyplot as plt
import logging # TODO Configure logging
import numpy as np
import yaml
import cv2

from utils.data import ADE20KDataset
from utils.utilsSAM import post_processing
from models.alphaClip import AlphaClip
from models.SAM import SAMSegmenter
from tqdm import tqdm


class Evaluator:
    def __init__(self,
                 sam: SAMSegmenter,
                 clip: AlphaClip,
                 loader: data.DataLoader,
                 device:str='cuda'):
        """
        :param sam: SAMSegmenter instance
        :param clip: AlphaClip instance
        :param loader: DataLoader instance
        :param device: device to use
        """
        self.sam = sam
        self.clip = clip
        self.loader = loader
        self.device = device

    def eval(self):
        loop = tqdm(self.loader, total=len(self.loader))
        print("-"*90)
        print("Starting evaluation")
        for i, batch in enumerate(loop):
            image = batch['image'].squeeze(0).to(self.device)
            vocabulary = batch['vocabulary']
            json_label = batch['label']

            masks = self.sam.predict_mask(image)
            images, masks = post_processing(masks, image, post_processing='none')
            logits = self.clip.classify(images, masks, vocabulary)
            predictions = torch.argmax(logits, dim=1)

            # assemble image
            # evaluate image

def main(args):
    sam = SAMSegmenter.from_args(args['sam'], device=args['device'])
    clip = AlphaClip.from_args(args['clip'], device=args['device'])
    loader = data.DataLoader(
        ADE20KDataset(
            args['root'], 
            transform=clip.preprocess,
            vocabulary='image_caption',
            ), batch_size=1, shuffle=False)
    
    evaluator = Evaluator(sam, clip, loader, device=args['device'])
    evaluator.eval()



if __name__ == '__main__':
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    
    with open('configs/sam_cfg.yaml', 'r') as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
    
    main(args)