import torch
import os
import glob
import pickle
from itertools import chain
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from lavis.models import load_model_and_preprocess

ROOT = 'datasets/ADE20K_2021_17_01/'

class ADE20KDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        
        with open('datasets/ADE20K_2021_17_01/image_paths.txt') as f:
            self.image_paths = [x.strip() for x in f.read().splitlines()]
        
        print("Loaded dataset with {} images".format(len(self.image_paths)))
       

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

def eval_loop(dataloader, model):
    captions = []
    for idx, batch in enumerate(tqdm(dataloader, desc="Generating captions", unit="batch")):
        captions.extend(model.generate({"image": batch.to(device)}, use_nucleus_sampling=True, num_captions=10))
        if idx == 100:
            print(captions[-10:])

    vocabulary = set(chain(*[cap.split(" ") for cap in captions]))
    
    # dump in apickle file
    with open(os.path.join(ROOT , 'captions_val/vocabulary.pkl'), 'wb') as f:
        pickle.dump(vocabulary, f)
    with open(os.path.join(ROOT , 'captions_val/captions.pkl'), 'wb') as f:
        pickle.dump(captions, f)

    return vocabulary
            
if __name__ == '__main__':
    device = 'cuda'
    dataset_root = '/datasets/ADE20K_2021_17_01/images'
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
    ])
    data = ADE20KDataset(dataset_root, transform=transform)
    # we associate a model with its preprocessors to make it easier for inference.
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip_caption", model_type="large_coco", is_eval=True, device=device
    )
    # uncomment to use base model
    # model, vis_processors, _ = load_model_and_preprocess(
    #     name="blip_caption", model_type="base_coco", is_eval=True, device=device
    # )
    dataloader = DataLoader(data, batch_size=8, shuffle=False, num_workers=4)

    eval_loop(dataloader, model)