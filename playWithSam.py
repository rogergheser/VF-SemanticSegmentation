
import sys
import pickle

from utils.utilsSAM import *
from models.SAM import SAMSegmenter
from models.alphaClip import AlphaClip


from datasets.dataset_vars import (
        COCO_CATEGORIES, 
        ADE20K_SEM_SEG_FULL_CATEGORIES as ADE20K_CATEGORIES
)

PATH_WEIGHT_SAM = 'checkpoints/sam_vit_h_4b8939.pth'
PATH_WEIGHT_ACLIP = 'checkpoints/clip_b16_grit+mim_fultune_4xe.pth'





if __name__ == "__main__":
    image = cv2.imread('datasets/ADE20K_2021_17_01/images/ADE/training/cultural/apse__indoor/ADE_train_00001472.jpg')
    
    segmenter = SAMSegmenter(model_type='vit_h', weight_path = PATH_WEIGHT_SAM, device='cuda')

    classifier = AlphaClip(model_type='ViT-B/16', weight_path = PATH_WEIGHT_ACLIP, device='cuda')


    path_files = 'datasets/subsetADE.txt'


    path_images = read_line_file(path_files)

    methods = ["blurred_masks", "red_circle_masks", "black_background_masks", "bbox_masks", "none"]
    vocabulary = take_vocabulary(dataset = COCO_CATEGORIES)

    results = segment_and_classify(segmenter, classifier, path_images, vocabulary, methods)

    # save results in a pikle file
    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)

    # load results from a pikle file
    with open('results.pkl', 'rb') as f:
        results = pickle.load(f)
    
    print(results)