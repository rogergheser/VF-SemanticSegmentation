import torch.utils.data as data
import glob
import json

from torchvision.transforms import Compose
from itertools import chain
from PIL import Image


class ADE20KDataset(data.Dataset):
    def __init__(self, root:str,
                 transform:Compose=None,
                 vocabulary:str='image_caption',
                 device:str='cuda'):
        """
        ADE20K dataset for image segmentation and captioning
        Vocabulary types:
            - ade_gt: ground truth segmentation labels
            - ade_caption: captions from the full ADE20K dataset
            - image_caption: captions generated from image using BLIP model
        :param root: root directory of the dataset
        :param transform: image preprocessing transformation
        :param vocabulary: vocabulary generation type
        :param device: device to use, default='cuda'
        """
        self.root = root
        self.transform = transform
        self.vocab_type = vocabulary
        self.vocabulary = None

        self.captioner = None
        self.cap_preprocess = None
        
        # with open('datasets/ADE20K_2021_17_01/image_paths.txt') as f:
        #     self.image_paths = [x.strip() for x in f.read().splitlines()]
        image_paths = glob.glob(f'{root}/**/*.jpg', recursive=True)
        label_paths = glob.glob(f'{root}/**/*.json', recursive=True)
        
        self.image_paths = image_paths
        self.label_paths = label_paths
        
        match self.vocab_type:
            case 'ade_gt':
                from datasets.dataset_vars import ADE20K_SEM_SEG_FULL_CATEGORIES as ADE20K
                self.vocabulary = [x['name'].split(", ")[0] for x in ADE20K]
            case 'ade_caption':
                import pickle
                try:
                    with open('datasets/ADE20K_2021_17_01/captions_val/vocabulary.pkl', 'rb') as f:
                        self.vocabulary = pickle.load(f)
                except:
                    raise FileNotFoundError('Could not find vocabulary.pkl')
            case 'image_caption':
                # define caption and create vocabulary on the fly
                from lavis.models import load_model_and_preprocess
                self.captioner, self.cap_preprocess, _ = load_model_and_preprocess(
                    name="blip_caption", model_type="large_coco", is_eval=True, device=device
                )
            case _:
                raise ValueError(f"Invalid vocabulary type: {self.vocab_type}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image_copy = image.copy()
        if self.transform:
            image = self.transform(image)
        
        with open(self.label_paths[idx]) as f:
            label = json.loads(f.read())
        
        if self.vocab_type == 'image_caption':
            assert self.captioner is not None, "Captioner not defined"
            _image = self.cap_preprocess['eval'](image_copy).unsqueeze(0)
            captions = self.captioner.generate({"image": _image.to('cuda')}, use_nucleus_sampling=True, num_captions=10)
            self.vocabulary = list(set(chain(*[cap.split(" ") for cap in captions])))
        
        sample = {
            'image': image,
            'vocabulary': self.vocabulary,
            'label': label
        }

        return sample