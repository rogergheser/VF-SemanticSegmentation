import torch
import torchvision
import torch.utils.data as data
import matplotlib.pyplot as plt
import logging # TODO Configure logging
import numpy as np
import yaml
import cv2
import os

from datasets.dataset_vars import (
    ADE20K_SEM_SEG_FULL_CATEGORIES as ADE20K_CATEGORIES
)
from utils.data import (
    ADE20KDataset,
    QualitativeDataset,
    Coco
)
from utils.utilsSAM import (
    post_processing,
    recompose_image
)
from torchvision import transforms as transform
from models.alphaClip import AlphaClip
from models.SAM import SAMSegmenter
from tqdm import tqdm

# evaluation using SAN evaluator
import sys
sys.path.append('/home/disi/VF-SemanticSegmentation/SAN')
from SAN.custom_evaluator import CustomSemSegEvaluator
from detectron2.modeling.meta_arch.build import build_model
from SAN.eval_net import setup
from detectron2.engine import default_argument_parser
class Evaluator:
    def __init__(self,
                 sam: SAMSegmenter,
                 clip: AlphaClip,
                 loader: data.DataLoader,
                 evaluator: CustomSemSegEvaluator,
                 device:str='cuda',
                 args: dict=None):
        """
        :param sam: SAMSegmenter instance
        :param clip: AlphaClip instance
        :param loader: DataLoader instance
        :param device: device to use
        """
        self.sam = sam
        self.clip = clip
        self.loader = loader
        self.evaluator = evaluator
        self.device = device
        self.save_results = args['save_results']
        self.overlay = args['overlay']
        self.ade_voc = {}
        self.new_label_idx = 0
        # for i, category in enumerate(ADE20K_CATEGORIES):
        #     keys = category["name"].split(", ")
        #     self.new_label_idx += 1
        #     for key in keys:
        #         if key not in self.ade_voc:
        #             self.ade_voc[key] = category["trainId"]

    def eval(self):
        os.makedirs('overlay', exist_ok=True)
        loop = tqdm(self.loader, total=len(self.loader))
        print("-"*90)
        print("Starting evaluation")
        for i, batch in enumerate(loop):
            image = batch['image'].squeeze(0).to(self.device)
            vocabulary = batch['vocabulary']
            json_label = batch['label']

            masks = self.sam.predict_mask(image)

            images, masks = post_processing(masks, image.type(torch.float32), post_processing='none')
            logits = self.clip.classify(images, masks, vocabulary)
            predictions = torch.argmax(logits, dim=1)
            text_predictions = [vocabulary[pred.item()] for pred in predictions]
            semseg = self.add_labels(image, text_predictions, masks)
            if self.save_results:
                overlay = recompose_image(image.cpu().numpy(), masks, overlay=self.overlay)
                cv2.imwrite(f'overlay/{i}.png', overlay.transpose(1, 2, 0))
            # TODO: evaluate image
            output = [{'sem_seg': logits}]
            self.evaluator.process(inputs=batch, outputs=output)

    def add_labels(self, image, text_predictions, masks):
        for text in text_predictions:
            if text not in self.ade_voc:
                self.ade_voc[text] = self.new_label_idx
                self.new_label_idx += 1
        newshape = image.shape
        newshape = (1, newshape[1], newshape[2])
        semseg = torch.zeros(newshape, dtype=torch.int32)
        # semseg size is (1, W, H)
        # mask['segmentation'] shape is (W x H)
        for text, mask in zip(text_predictions, masks):
            semseg[:, mask['segmentation']] = self.ade_voc[text]

        return semseg
    
def get_san_model():
    detectron_args = default_argument_parser().parse_args()

    detectron_args.config_file = 'SAN/configs/san_clip_vit_res4_coco.yaml'
    detectron_args.eval_only = True
    detectron_args.opts = ['OUTPUT_DIR', 'output/ade20k_full_SAM_eval', 'MODEL.WEIGHTS', 'san_vit_b_16.pth', 'DATASETS.TEST', "('ade20k_full_sem_seg_val',)"]
    cfg = setup(detectron_args)
    san_model = build_model(cfg)

    return san_model, cfg

def main(dataset, args):
    sam = SAMSegmenter.from_args(args['sam'], device=args['device'])
    clip = AlphaClip.from_args(args['clip'], device=args['device'])
    loader = data.DataLoader(
        dataset, 
        batch_size=args['dataloader']['batch_size'],
        shuffle=args['dataloader']['shuffle'],)
    
    san_model, san_cfg = get_san_model()
    quantitative_evaluator=CustomSemSegEvaluator(san_model, args['dataset']['name'], False, san_cfg.OUTPUT_DIR)

    evaluator = Evaluator(sam, clip, loader, quantitative_evaluator, device=args['device'], args=args)
    evaluator.eval()


if __name__ == '__main__':
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    
    with open('configs/sam_cfg.yaml', 'r') as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
    transform = transform.Compose([
        # transform.Resize((args['dataset']['resize'], args['dataset']['resize'])),
        transform.PILToTensor(),
    ])

    dataset_name_to_class = {
        'qualitative': QualitativeDataset,
        'ade20k_full_sem_seg_val': ADE20KDataset,
        'coco' : Coco
    }
    
    dataset_name = args['dataset']['name']
    dataset_class = dataset_name_to_class[dataset_name]
    dataset = dataset_class.from_args(args, transform)

    main(dataset, args)