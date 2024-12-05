import os
import cv2
import copy
import numpy as np
from typing import List


def import_vocabulary(dataset) -> List[str]:
    default_voc = []
    for c in dataset:
        tmp = c["name"] + ", a"
        default_voc.append(c["name"].split(", ")[0])

    return default_voc


def take_vocabulary( dataset=None, add_words=None):

    if dataset is not None:
        vocabulary = import_vocabulary(dataset)

    if add_words is not None:
        add_words = list(set([v.lower().strip() for v in add_words]))
        # remove invalid vocabulary
        add_words = [v for v in add_words if v != ""]

        vocabulary = add_words + [c for c in vocabulary if c not in add_words]

    return vocabulary


def save_masks(masks, output_dir):
        
    os.makedirs(output_dir, exist_ok=True)

    for i, mask_dict in enumerate(masks):
        # Extract the segmentation mask
        mask = mask_dict['segmentation']
        
        # Save the binary mask as an image
        mask_path = os.path.join(output_dir, f"mask_{i}.png")
        cv2.imwrite(mask_path, mask.astype('uint8') * 255)
        print(f"Saved mask to {mask_path}")


def add_padding(bbox, image_shape, padding_p):
    """
    :param bbox: [x, y, w, h]
    :param image_shape: (height, width, channels)
    :param padding_p: padding percentage
    """
    x, y, w, h = bbox
    im_height, im_width, _ = image_shape
    padding = int(padding_p * max(w, h))

    y1 = max(0, y - padding)
    x1 = max(0, x - padding)
 
    new_height = im_height - y1 if h + 2 * padding > im_height else h + 2 * padding
    new_width = im_width - x1 if w + 2 * padding > im_width else w + 2 * padding
    
    if new_height == new_width:
        return x1, y1, new_height, new_width
    elif new_height > new_width:
        diff = new_height - new_width
        x1 = max(0, x1 - diff // 2)
        new_width = new_height
        return x1, y1, new_height, new_width
    else:
        diff = new_width - new_height
        y1 = max(0, y1 - diff // 2)
        new_height = new_width
        return x1, y1, new_height, new_width


def post_processing(masks, image, post_processing='blurred_masks'):
    
    masks_copy = copy.deepcopy(masks)
    if post_processing == 'blurred_masks':
        images = blurred_masks(masks_copy, image)
    elif post_processing == 'red_circle_masks':
        images = red_circle_masks(masks_copy, image)
    elif post_processing == 'bbox_masks':
        images, masks_copy = bbox_masks(masks_copy, image)
    elif post_processing == 'black_background_masks':
        images = black_background_masks(masks_copy, image)
    else:
        print("no post-processing")
        images = [image for _ in masks]
        

    return images, masks_copy


def blurred_masks(masks, image):
    # apply a Gaussian blur to the entire image where the mask is 0
    images = []

    for i, mask_dict in enumerate(masks):
        binary_mask = mask_dict['segmentation']

        binary_mask = binary_mask.astype(np.uint8)
        
        inverted_mask = 1 - binary_mask
        
        blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
        
        combined_image = image.copy()
        for c in range(image.shape[2]):  # Iterate over color channels
            combined_image[:, :, c] = (
                image[:, :, c] * binary_mask +
                blurred_image[:, :, c] * inverted_mask
            )

        images.append(combined_image)

    return images


def red_circle_masks(masks, image):
    images = []

    for i, mask_dict in enumerate(masks):
        x, y, w, h = mask_dict['bbox']

        center_x = int(x + w / 2)
        center_y = int(y + h / 2)

        width_len = int(w * 0.75)
        height_len = int(h * 0.75)

        processed_image = image.copy()

        cv2.ellipse(processed_image, (center_x, center_y), (width_len, height_len), 0, 
                          0, 360, (0, 0, 255), thickness=4) 

        # cv2.circle(processed_image, (center_x, center_y), radius, (0, 0, 255), thickness=4)

        images.append(processed_image)

    return images


def bbox_masks(masks, image):
    images = []

    for i, mask_dict in enumerate(masks):
        x, y, w, h = mask_dict['bbox']
        x, y, h, w = add_padding((x, y, w, h), image.shape, 0.15)

        processed_image = image.copy()
        processed_image = processed_image[y:y+h, x:x+w]
        masks[i]['segmentation'] = masks[i]['segmentation'][y:y+h, x:x+w]

        images.append(processed_image)
    
    return images, masks


def black_background_masks(masks, image):

    images = []

    for i, mask_dict in enumerate(masks):
        binary_mask = mask_dict['segmentation']
        binary_mask = binary_mask.astype(np.uint8)

        processed_image = image.copy()
        processed_image = processed_image * binary_mask[:, :, np.newaxis]

        images.append(processed_image)
    
    return images


def read_line_file(path_files):
    ret_lines = []
    with open(path_files, 'r') as file:
        for line in file:
            ret_lines.append("../" + line.strip() + ".jpg")
    
    return ret_lines


def segment_and_classify(segmenter, classifier, path_images, vocabulary, methods):
    imgs = []
    segmentations = []
    results_logits = []

    for path_image in path_images:
        print(f"Processing image {path_image}")

        image = cv2.imread(path_image)

        # Segment Image
        masks = segmenter.predict_mask(image)

        masks_sam_copy = copy.deepcopy(masks)

        result_logit = {}

        for method in methods:

            #Post Processing
            images, masks_sam_copy = post_processing(masks_sam_copy, image, post_processing=method)

            # Classify Mask
            logits = classifier.classify_mask(images, masks_sam_copy, vocabulary, flagUseAlpha = True)

            result_logit[method] = logits
        
        imgs.append(image)
        segmentations.append(masks)
        results_logits.append(result_logit)
        

    results = {
        'images': imgs,
        'segmentations': segmentations,
        'logits': results_logits
    }

    return results

