import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

### arguments

device = 'mps'
sam_checkpoint = ''
image = cv2.imread('data/dog.jpg')
model_type = 'vit_h'



def pipeline():
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)
    mask_generator2 = SamAutomaticMaskGenerator(
        model = sam,
        points_per_side = 32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
        )


    masks = mask_generator.generate(image)
    # masks is a list of dictionaries, each dictionary contains the following:
    # 'segmentation': the mask
    # 'area': the area of the mask in pixels
    # 'bbox': the bounding box of the mask
    # 'predicted_iou' : the model's own prediction for the quality of the mask
    # 'point_coords' : the sampled input point that generated this mask
    # 'stability_score' : an additional measure of mask quality
    # 'crop_box' : the crop of the image used to generate this mask in XYWH format

    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show() 