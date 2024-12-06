import math
import torch
import alpha_clip
import numpy as np
from PIL import Image
from torchvision import transforms


class AlphaClip:
    def __init__(self,
                 model_type: str,
                 weight_path: str,
                 device: str = 'cuda'):
        """
        Wrapper for the AlphaCLIP model.
        :param model_type: the model type to load
        :param weight_path: the path to the model weights
        :param device: the device to use, default='cuda'
        """
        self.model_type = model_type
        self.weight_path = weight_path
        self.device = device
        self.model, self.preprocess = alpha_clip.load(self.model_type, self.weight_path, device=self.device)
        
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize((224, 224)),
            transforms.Normalize(0.5, 0.26)
        ])

    @classmethod
    def from_args(cls, args, device='cuda'):
        """
        Load AlphaCLIP from the arguments specified in the config.
        """
        return cls(model_type=args['model_type'], weight_path=args['weight_path'], device=device)

    def prepare_batch(self,
                      images, # ?? What type??
                      masks,  # ?? What type??
                      vocabulary : list[str],
                      flagUseAlpha : bool = True):
        """
        Prepares batches for the AlphaCLIP model.
        :param images: list of images

        """
        imgs = []
        alphas = []
        prompts = self.prompts_from_vocab(vocabulary)
        tokenized_prompts = alpha_clip.tokenize(prompts)

        for mask, image in zip(masks, images):        
            binary_mask = mask['segmentation']
            
            if flagUseAlpha:
                alpha = self.mask_transform((binary_mask * 255).astype(np.uint8))
            else:
                alpha = torch.ones(1, 224, 224)
                
            alpha = alpha.half().cuda()
            
            image = Image.fromarray(image)
            image = self.preprocess(image).half()

            imgs.append(image)
            alphas.append(alpha)

        batch = {
            'image': torch.stack(imgs),
            'alpha': torch.stack(alphas),
            'text': tokenized_prompts
        }
            
        return batch

    def prompts_from_vocab(self, vocabulary: list[str]):
        return [f"a photo of a {v}" for v in vocabulary]
    
    def prepare_mask(self, mask : list[dict]):
        binary_mask = mask['segmentation']
        alpha = self.mask_transform((binary_mask * 255).astype(np.uint8))
        alpha = alpha.half().cuda()
        return alpha

    def classify(self,
                 images : list[torch.Tensor], 
                 masks : list[dict], 
                 vocabulary : list[str]):
        """
        Classification function used in the pipeline.
        :param images: list of images
        :param masks: list of masks
        :param vocabulary: list of vocabulary to be tokenized
        """
        preprocess = transforms.Compose([self.preprocess.transforms[0],
                                        self.preprocess.transforms[1],
                                        self.preprocess.transforms[-1]])
        imgs = torch.stack([preprocess(image).half().cuda() for image in images])
        alphas = torch.stack([self.prepare_mask(mask) for mask in masks])
        text = alpha_clip.tokenize(self.prompts_from_vocab(vocabulary)).to(self.device)

        with torch.no_grad():
            image_features = self.model.visual(imgs, alphas)
            text_features = self.model.encode_text(text) 

        # normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits_scale = math.exp(1.2) 
        logits = (logits_scale * 100.0 * image_features @ text_features.T).softmax(dim=-1)

        return logits


    def classify_mask(self, images, masks, vocabulary, flagUseAlpha = True):
        """
        Mask classifier function used in the notebook. Not used in the pipeline.
        :param images: list of images
        :param masks: list of masks
        :param vocabulary: list of vocabulary to be tokenized
        :param flagUseAlpha: flag to use alpha or not
        """
        batch = self.prepare_batch(images, masks, vocabulary, flagUseAlpha)
        cropped_img, alpha, tokenized_prompts = batch['image'], batch['alpha'], batch['text']
        
        cropped_img = cropped_img.to(self.device)
        alpha = alpha.to(self.device)
        text = tokenized_prompts.to(self.device)

        with torch.no_grad():
            image_features = self.model.visual(cropped_img, alpha)
            text_features = self.model.encode_text(text) 

        # normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits_scale = math.exp(1.2) 
        logits = (logits_scale * 100.0 * image_features @ text_features.T).softmax(dim=-1)

        return logits
