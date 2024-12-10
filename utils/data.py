import torch.utils.data as data
import glob
import json
import os
from torchvision.transforms import Compose
from itertools import chain
from PIL import Image

class QualitativeDataset(data.Dataset):
    def __init__(self,
                 root:str,
                 transform:Compose=None,
                 vocabulary:str='image_caption',
                 device:str='cuda'):
        """
        Qualitative dataset for image segmentation
        :param root: root directory of the dataset
        :param transform: image preprocessing transformation
        """
        self.root = root
        self.transform = transform
        self.vocab_type = vocabulary
        self.vocabulary = None

        self.captioner = None
        self.cap_preprocess = None

        with open('datasets/subsetADE.txt') as f:
            self.paths = [x.strip() for x in f.read().splitlines()]
        self.image_paths = [x + '.jpg' for x in self.paths]
        self.label_paths = [x + '.json' for x in self.paths]

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
    
    @classmethod
    def from_args(cls, args:dict, transform: Compose=None):
        return cls(
            root=args['dataset']['root'],
            transform=transform,
            vocabulary=args['dataset']['vocabulary'],
            device=args['device']
        )
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

class ADE20KDataset(data.Dataset):
    def __init__(self,
                 root:str,
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
        # label_paths = glob.glob(f'{root}/**/*.json', recursive=True)
        
        self.image_paths = image_paths
        # self.label_paths = label_paths
        
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

    @classmethod
    def from_args(cls, args: dict, transform: Compose=None):
        return cls(
            root=args['dataset']['root'],
            transform=transform,
            vocabulary=args['dataset']['vocabulary'],
            device=args['device']
        )
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image_copy = image.copy()
        if self.transform:
            image = self.transform(image)
        
        # with open(self.label_paths[idx]) as f:
        #     label = json.loads(f.read())
        
        if self.vocab_type == 'image_caption':
            assert self.captioner is not None, "Captioner not defined"
            _image = self.cap_preprocess['eval'](image_copy).unsqueeze(0)
            captions = self.captioner.generate({"image": _image.to('cuda')}, use_nucleus_sampling=True, num_captions=10)
            self.vocabulary = list(set(chain(*[cap.split(" ") for cap in captions])))
        
        sample = {
            'image': image,
            'vocabulary': self.vocabulary,
            'label': 'maskemerda',
            'file_name': os.path.join(os.getcwd(), image_path)
        }

        return sample
    
class Coco(data.Dataset):
    def __init__(self,
                 root:str,
                 transform:Compose=None,
                 vocabulary:str='image_caption',
                 device:str='cuda'):
        """
        Coco dataset for image segmentation and captioning
        Vocabulary types:
            - coco_gt: ground truth segmentation labels
            - coco_caption: captions from the full coco dataset
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

        self.image_paths = glob.glob(f'{root}/val2017/*.jpg', recursive=False) # All coco files are in the same dir
        with open(os.path.join(self.root, 'annotations/instances_val2017.json')) as f:
            self.labels = json.load(f)
        with open(os.path.join(self.root, 'annotations/captions_val2017.json')) as f:
            self.captions = json.load(f)

    @classmethod
    def from_args(cls, args: dict, transform: Compose=None):
        return cls(
            root=args['dataset']['root'],
            transform=transform,
            vocabulary=args['dataset']['vocabulary'],
            device=args['device']
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_id = int(self.image_paths[idx].rsplit('/', 1)[1].split('.')[0])
        
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image_copy = image.copy()
        if self.transform:
            image = self.transform(image)

        label = {}
        label['masks'] = [label for label in self.labels['annotations'] if label['image_id'] == 139]
        label['ids'] = self.labels['categories']
        

        match self.vocab_type:
            case 'coco_gt':
                self.vocabulary = [label['name'] for label in self.labels['categories']]
            case 'coco_caption':
                captions = [caption for caption in self.captions['annotations'] if caption['image_id']==image_id]
                self.vocabulary = list(set(chain(*[caption['caption'].lower().split(" ") for caption in captions])))
            case 'image_caption':
                # define caption and create vocabulary on the fly
                from lavis.models import load_model_and_preprocess
                self.captioner, self.cap_preprocess, _ = load_model_and_preprocess(
                    name="blip_caption", model_type="large_coco", is_eval=True, device=self.device
                )
                _image = self.cap_preprocess['eval'](image_copy).unsqueeze(0)
                captions = self.captioner.generate({"image": _image.to('cuda')}, use_nucleus_sampling=True, num_captions=10)
                self.vocabulary = list(set(chain(*[cap.split(" ") for cap in captions])))
            case _:
                raise ValueError(f"Invalid vocabulary type: {self.vocab_type}")
            
        sample = {
            'image': image,
            'vocabulary': self.vocabulary,
            'label': label,
            'file_name': os.path.join(os.getcwd(), image_path)
        }

        return sample

if __name__ == '__main__':
    # Test the dataset
    from torchvision import transforms as transform

    transform = transform.Compose([
        transform.Resize((256, 256)),
        transform.ToTensor()
    ])

    dataset = Coco.from_args({
        'dataset': {
            'root': 'datasets/coco',
            'vocabulary': 'image_caption'
        },
        'device': 'cuda'
    }, transform=transform)