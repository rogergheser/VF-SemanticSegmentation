import cv2
import numpy as np
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
from torchvision.transforms import ToPILImage

class SAMSegmenter:
    def __init__(self,
                 model_type: str,
                 weight_path: str,
                 device: str = 'cuda'):
        """
        SamSegmenter class to predict segmentation masks using SAM model.
            model_type: e.g. vit_l, vit_h ecc...
        :param model_type: model type to load
        :param weight_path: path to the weight file
        :param device: device to use, default='cuda'
        """
        self.model_type = model_type
        self.weight_path = weight_path
        self.device = device
        self.model = self._load_model()

    @classmethod
    def from_args(cls, args, device='cuda'):
        """
        Load SAM from the arguments specified in the config.
        """
        return cls(model_type=args['model_type'], weight_path=args['weight_path'], device=device)
    
    def _load_model(self):

        print(f"Loading weight for SAM")
        sam = sam_model_registry[self.model_type](checkpoint=self.weight_path)
        sam.to(self.device)

        return sam

    def predict_mask(self, image, points=None, labels=None):
        """
        Predict segmentation masks for the input image.
        """
        if points is not None and labels is not None:
            predictor = SamPredictor(self.model)
            predictor.set_image(image)
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
                min_mask_region_area=100,  # Requires open-cv to run post-processing
            )
            img = np.transpose(image.cpu().numpy(), (1, 2, 0))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            masks = mask_generator.generate(img)
            print("Generated {} masks".format(len(masks)))
            
            return masks









