import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import time
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    
    cv2.imwrite('images/mask.jpg', img)
    
### arguments


def pipeline(device, sam_checkpoint, image, model_type):
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

    return masks

if __name__ == '__main__':
    
    device = 'cuda'
    sam_checkpoint = './checkpoints/sam_vit_h_4b8939.pth'
    image = cv2.imread('images/dog.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(20,20))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    time.sleep(2)
    exit(1)
    model_type = 'vit_h'

    masks = pipeline(device, sam_checkpoint, image, model_type)
    print('done')