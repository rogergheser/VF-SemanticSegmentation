import matplotlib.pyplot as plt
from utilsSAM import *
from SAM import SAMSegmenter
from alphaClip import AlphaClip


from datasets.dataset_vars import (
        COCO_CATEGORIES, 
        ADE20K_SEM_SEG_FULL_CATEGORIES as ADE20K_CATEGORIES
)

PATH_WEIGHT_SAM = 'checkpoints/sam_vit_h_4b8939.pth'
PATH_WEIGHT_ACLIP = 'checkpoints/clip_b16_grit+mim_fultune_4xe.pth'

if __name__ == "__main__":
    image = cv2.imread('datasets/ADE20K_2021_17_01/images/ADE/training/cultural/apse__indoor/ADE_train_00001472.jpg')
    
    segmenter = SAMSegmenter(model_type='vit_h', weight_path = PATH_WEIGHT_SAM)
    masks = segmenter.predict_mask(image)

    images, masks = post_processing(masks, image, post_processing='black_background_masks')
    vocabulary = take_vocabulary(dataset = COCO_CATEGORIES)
    
    classifier = AlphaClip(model_type='ViT-B/16', weight_path = PATH_WEIGHT_ACLIP)
    logits = classifier.classify_mask(images, masks, vocabulary, flagUseAlpha = False)
 

    predictions = logits.argmax(dim=-1)
    values, indices = logits.cpu().topk(5)
    for dim in range(len(values)):
        print("\n\nTop 5 predictions for mask ", dim)
        for value, index in zip(values[dim], indices[dim]):
            print(f"\t{vocabulary[index]}: {value.item()}")
