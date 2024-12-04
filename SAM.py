import os
import cv2
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry

class SAMSegmenter:

    def __init__(self, model_type: str, weight_path: str, device: str = 'cuda'):
        """
            model_type: ex vit_l, vit_h ecc...
        """
        self.model_type = model_type
        self.weight_path = weight_path
        self.device = device
        self.model = self._load_model()

    def _load_model(self):

        print(f"Loading weight for SAM")
        sam = sam_model_registry[self.model_type](checkpoint=self.weight_path)
        sam.to(self.device)

        return sam

    def predict_mask(self, image, points=None, labels=None):
        """
        Predict segmentation masks for the input image.
        """

        predictor = SamPredictor(self.model)
        predictor.set_image(image)

        if points is not None and labels is not None:
            masks, scores, logits = predictor.predict(point_coords=points, point_labels=labels, multimask_output=True)
            return masks, scores, logits
        else:
            mask_generator = SamAutomaticMaskGenerator(
                model=self.model,
                points_per_side=20,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=0,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=400,  # Requires open-cv to run post-processing
            )
            masks = mask_generator.generate(image)
            return masks

    def save_masks(self, masks, output_dir):
        
        os.makedirs(output_dir, exist_ok=True)

        for i, mask_dict in enumerate(masks):
            # Extract the segmentation mask
            mask = mask_dict['segmentation']
            
            # Save the binary mask as an image
            mask_path = os.path.join(output_dir, f"mask_{i}.png")
            cv2.imwrite(mask_path, mask.astype('uint8') * 255)
            print(f"Saved mask to {mask_path}")

    def blurred_masks():
        print("post processing: blur masks")

    def red_circle_masks():
        print("post processing: create a red circle in the image")

    def bbox_masks():
        print("post processing: bbox masks")
    
    def black_background_masks():
        print("post processing: black background masks")







