import math
import torch
import alpha_clip
import numpy as np
from PIL import Image
from torchvision import transforms


class AlphaClip:

    def __init__(self, model_type: str, weight_path: str, device: str = 'cuda'):

        self.model_type = model_type
        self.weight_path = weight_path
        self.device = device
        self.model, self.preprocess = alpha_clip.load(self.model_type, self.weight_path, device=self.device)
        
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize((224, 224)),
            transforms.Normalize(0.5, 0.26)
        ])


    def prepare_batch(self, images, masks, vocabulary, flagUseAlpha):
        imgs = []
        alphas = []
        prompts = self.prompts_from_vocab(vocabulary)
        tokenized_prompts = alpha_clip.tokenize(prompts)

        for mask, image in zip(masks, images):        
            binary_mask = mask['segmentation']
            
            if flagUseAlpha:
                alpha = self.mask_transform((binary_mask * 255).astype(np.uint8))
            else:
                alpha = torch.ones(224, 224)
                
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


    def classify_mask(self, images, masks, vocabulary, flagUseAlpha = True):

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

    def prompts_from_vocab(self, vocabulary):
        return [f"a photo of a {v}" for v in vocabulary]