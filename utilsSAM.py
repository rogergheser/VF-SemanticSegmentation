import os
import cv2
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

    if post_processing == 'blurred_masks':
        images = blurred_masks(masks, image)
    elif post_processing == 'red_circle_masks':
        images = red_circle_masks(masks, image)
    elif post_processing == 'bbox_masks':
        images, masks = bbox_masks(masks, image)
    elif post_processing == 'black_background_masks':
        images = black_background_masks(masks, image)
    else:
        print("Invalid post processing method")

    return images, masks


def blurred_masks(masks, image):
    # apply a Gaussian blur to the entire image where the mask is 0
    print("--[INFO] post processing: blur masks--")
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
    print("--[INFO] post processing: create an empty red circle in the image--")
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
    print("--[INFO] post processing: bbox masks--")
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
    print("--[INFO] post processing: black background masks--")

    images = []

    for i, mask_dict in enumerate(masks):
        binary_mask = mask_dict['segmentation']
        binary_mask = binary_mask.astype(np.uint8)

        processed_image = image.copy()
        processed_image = processed_image * binary_mask[:, :, np.newaxis]

        images.append(processed_image)
    
    return images