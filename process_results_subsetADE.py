from utils.utilsSAM import *
from segment_classify_subsetADE import process_and_save_image_seg_class

from datasets.dataset_vars import (
        COCO_CATEGORIES, 
        ADE20K_SEM_SEG_FULL_CATEGORIES as ADE20K_CATEGORIES
)


if __name__ == "__main__":

    # process_and_save_image_seg_class()

    results = read_pickle('resultsSAM/results_subsetADE.pkl')

    vocabulary = take_vocabulary(dataset = COCO_CATEGORIES)

    path_files = 'datasets/subsetADE.txt'

    path_images = read_line_file(path_files, additional_path="")

    # create a folder to save the results of images
    path_dir = 'resultsSAM/results_subsetADE_images'
    os.makedirs(path_dir, exist_ok=True)
    

    for image, seg, logit, path_image in zip(results['images'], results['segmentations'], results['logits'], path_images):

        name_image = path_image.split('/')[-1].split('.')[0]
        path_dir_image = os.path.join(path_dir, os.path.basename(name_image))
        os.makedirs(path_dir_image, exist_ok=True)
        cv2.imwrite(os.path.join(path_dir_image, 'original.jpg'), image)

        path_gt = path_image.split('.')[0] + '_seg.png'
        image_gt = cv2.imread(path_gt)
        cv2.imwrite(os.path.join(path_dir_image, 'gt.png'), image_gt)
        
        filtered_seg, _ = filter_largest_masks(seg)

        overlay_image = recompose_image(image, filtered_seg)

        cv2.imwrite(os.path.join(path_dir_image, 'SAM_seg.jpg'), overlay_image)

        path_annotation = os.path.join(path_dir_image, 'SAM_seg_annotated')
        os.makedirs(path_annotation, exist_ok=True)
        path_post_processing = os.path.join(path_annotation, 'post_processing')
        os.makedirs(path_post_processing, exist_ok=True)

        for method in logit.keys():

            predictions = logit[method].argmax(axis=-1)

            filtered_seg, filtered_prediction = filter_largest_masks(seg, predictions)
            filtered_seg_copy = copy.deepcopy(filtered_seg)

            images, filtered_seg_copy = post_processing(filtered_seg_copy, image, post_processing='_'.join(method.rsplit('_', 1)[:-1]))

            path_post_processing_method = os.path.join(path_post_processing, f'{method}')
            os.makedirs(path_post_processing_method, exist_ok=True)

            for i, image_post in enumerate(images):
                cv2.imwrite(os.path.join(path_post_processing_method, f'seg_{i}_post_processing.jpg'), image_post)
            
            annotated_overlay = annotate_predictions_on_image(overlay_image, filtered_seg, filtered_prediction, vocabulary)

            cv2.imwrite(os.path.join(path_annotation, f'{method}.jpg'), annotated_overlay)

