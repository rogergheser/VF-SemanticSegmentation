
# import sys
import pickle
import torch

from utils.utilsSAM import *
from models.SAM import SAMSegmenter
from models.alphaClip import AlphaClip


from datasets.dataset_vars import (
        COCO_CATEGORIES, 
        ADE20K_SEM_SEG_FULL_CATEGORIES as ADE20K_CATEGORIES
)

PATH_WEIGHT_SAM = 'checkpoints/sam_vit_h_4b8939.pth'
PATH_WEIGHT_ACLIP = 'checkpoints/clip_b16_grit+mim_fultune_4xe.pth'



def process_and_save_image_seg_class():
    image = cv2.imread('datasets/ADE20K_2021_17_01/images/ADE/training/cultural/apse__indoor/ADE_train_00001472.jpg')
    
    segmenter = SAMSegmenter(model_type='vit_h', weight_path = PATH_WEIGHT_SAM, device='mps')

    classifier = AlphaClip(model_type='ViT-B/16', weight_path = PATH_WEIGHT_ACLIP, device='mps')


    path_files = 'datasets/subsetADE.txt'


    path_images = read_line_file(path_files, additional_path="")

    methods = ["blurred_masks", "red_circle_masks", "black_background_masks", "bbox_masks", "none"]
    vocabulary = take_vocabulary(dataset = COCO_CATEGORIES)

    results = segment_and_classify(segmenter, classifier, path_images, vocabulary, methods)

    # Save results to a pickle file in CPU-compatible format
    with open('resultsSAM/results_subsetADE.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Load results back on CPU
    with open('resultsSAM/results_subsetADE.pkl', 'rb') as f:
        results = pickle.load(f)

    # Print results
    print(results)



if __name__ == "__main__":
    
    process_and_save_image_seg_class()
