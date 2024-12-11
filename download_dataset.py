import shutil
import subprocess
import yaml
import os
import zipfile
import tarfile
import requests
from abc import ABC, abstractmethod
from tqdm import tqdm

####################################################################
# TODO Create an unzip function to avoid code repetition
# TODO Add an OUT field in YAML to check if the file is already downloaded
####################################################################

CONFIG_FILE = 'datasets.yaml'
POSTPROCESS = False

def field_exists(field: dict)->bool:
    """
    Checks if the field of the configuration files already exists.
    :param cfg: dictionary with the configuration of the dataset.
    Should be passed as cfg[dataset] e.g. cfg['COCO'].
    """
    target_file = os.path.join(field['dir'], os.path.basename(field['url']).split('.')[0])
    extracted_file = os.path.join(field['dir'], os.path.basename(field['url']).split('.')[0])
    if not os.path.exists(target_file):
        print('Current directory:\n{}'.format(os.getcwd()))
        print('Field {} does not exist'.format(field))
        return False
    return True

def download_file(url: str, destination: str):
    """
    Downloads a file from a given URL to a specified destination, showing a progress bar.
    :param url: string, URL of the file to download
    :param destination: string, destination path to save the file
    """
    target_file = os.path.join(destination, os.path.basename(url))
    # Create the directory if it doesn't exist
    if not os.path.exists(os.path.dirname(destination)):
        print(f"Creating directory: {os.path.dirname(destination)}")
        os.makedirs(os.path.dirname(destination))

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        # Send the HTTP GET request
        with requests.get(url, headers=headers, stream=True) as response:
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Get the total file size from the Content-Length header (if available)
            total_size = int(response.headers.get("Content-Length", 0))

            # Initialize the progress bar
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as progress_bar:
                # Open the destination file in write-binary mode
                with open(target_file, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=65536):  # Default chunk size
                        if chunk:  # Write only non-empty chunks
                            file.write(chunk)
                            progress_bar.update(len(chunk))  # Update progress bar

        print(f"File downloaded successfully to {destination}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")

class Dataset(ABC):
    """
    Abstract class for downloading datasets. 
    To extend this class, the _download_dataset method must be implemented.
    """
    def __init__(self, cfg: dict, download:bool=True, postprocess:bool=False):
        """
        :param cfg: dictionary with the configuration of the dataset.
        Should be passed as cfg[dataset] e.g. cfg['COCO'].
        :param download: boolean, whether to download the dataset or not.
        """
        self.cfg = cfg
        self.download = download

        if download:
            self._download_dataset()
        else:
            raise ValueError('Dataset not found')

    @abstractmethod
    def _download_dataset(self):
        pass


class COCO(Dataset):
    def _download_dataset(self):
        """
        Downloads files for the COCO dataset according to the provided configuration file.
        """
        for field in self.cfg:
            if field_exists(self.cfg[field]):
                print("Field {} already exists. Skipping download".format(field))
                continue

            link, dir = self.cfg[field]['url'], self.cfg[field]['dir']
            target_file = os.path.join(dir, os.path.basename(link).split('.')[0])
            download_file(link, dir)
            # All COCO download files are zip
            with zipfile.ZipFile(os.path.join(dir, link.split('/')[-1]), 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(dir))
            print("Successfully downloaded and extracted COCO:{}".format(field))
            os.remove(os.path.join(dir, link.split('/')[-1]))

class VOC2012(Dataset):
    def _download_dataset(self):
        """
        Downloads files for the VOC2012 dataset according to the provided configuration file.
        """
        if field_exists(self.cfg['trainval']):
            print("Field {} already exists. Skipping download".format(self.cfg['trainval']))
        else:
            link, dir = self.cfg['trainval']['url'], self.cfg['trainval']['dir']
            target_file = os.path.join(dir, os.path.basename(link).split('.')[0])
            download_file(link, dir)
            with tarfile.open(os.path.join(dir, link.split('/')[-1]), 'r') as tar:
                # Go over each member
                for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
                    # Extract member
                    tar.extract(member=member)
            os.remove(os.path.join(dir, link.split('/')[-1]))

            source_dir = 'VOCdevkit/VOC2012'
            for file in os.listdir(os.path.join(source_dir, 'JPEGImages')):
                shutil.move(os.path.join(source_dir, 'JPEGImages', file), dir)
            os.makedirs(self.cfg['val_file']['dir'], exist_ok=True)
            shutil.move(os.path.join(source_dir, 'ImageSets/Segmentation/val.txt'), self.cfg['val_file']['dir'])
            # TODO test VOCdevkit removal
            shutil.rmtree('VOCdevkit')
            print("Successfully downloaded and extracted part of VOC2012")

            url1, dir1 = self.cfg['SegClassAug']['url'], self.cfg['SegClassAug']['dir']
            url2, dir2 = self.cfg['train_file']['url'], self.cfg['train_file']['dir']
            print("Proceed by manually downloading url1 from dropbox and cloning url2 from gist\
                  \n\t{}\n\t\t{}\n\t\t{}\
                  \n\t{}\n\t\t{}\n\t\t{}".format(
                    'SegmentationClass', url1, dir1,
                    'train.txt', url2, dir2
                    )
                  )
            
        # ASSERT DOWNLOADED FILES ARE THERE OR PROMPT THE USER TO GET THEM

class PContext(Dataset):
    def _download_dataset(self):
        if field_exists(self.cfg['images']):
            print("Field {} already exists. Skipping download".format(self.cfg['images']))
        else:
            link, dir = self.cfg['images']['url'], self.cfg['images']['dir']
            target_file = os.path.join(dir, os.path.basename(link).split('.')[0])
            download_file(link, dir) # tar file
            with tarfile.open(os.path.join(dir, link.split('/')[-1]), 'r') as tar:
                # Go over each member
                for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
                    tar.extract(member=member, path=dir, filter='fully_trusted')
            os.remove(os.path.join(dir, link.split('/')[-1]))
            for file in os.listdir(os.path.join(dir, 'VOCdevkit/VOC2010/JPEGImages')):
                shutil.move(os.path.join(dir, 'VOCdevkit/VOC2010/JPEGImages', file), dir)
            shutil.rmtree(os.path.join(dir, 'VOCdevkit'))
            # TODO Move files to the correct directory
        
        if field_exists(self.cfg['labels']):
            print("Field {} already exists. Skipping download".format(self.cfg['labels']))
        else:
            link, dir = self.cfg['labels']['url'], self.cfg['labels']['dir']
            target_file = os.path.join(dir, os.path.basename(link).split('.')[0])
            download_file(link, dir) # json file

class ADEChallengeData2016(Dataset):
    def _download_dataset(self):
        if field_exists(self.cfg):
            print("Field {} already exists. Skipping download".format(self.cfg))
        else:
            link, dir = self.cfg['url'], self.cfg['dir']
            target_file = os.path.join(dir, os.path.basename(link))
            try:
                download_file(link, dir) # zip file
            except requests.exceptions.RequestException as e:
                print(f"Error downloading file: {e}")
                return
            with zipfile.ZipFile(target_file, 'r') as zip_ref:
                for member in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
                    zip_ref.extract(member=member, path=dir)
            os.remove(os.path.join(dir, link.split('/')[-1]))

class ADE20K(Dataset):
    def _download_dataset(self):
        if field_exists(self.cfg):
            print("Field {} already exists. Skipping download".format(self.cfg))
        else:
            link, dir = self.cfg['url'], self.cfg['dir']
            if link == 'link':
                print("ADE20K full requires authorisation for dataset download. Please download manually.")
                print("Alternatively, register and insert the correct link the datasets.yaml file.")
                print("Consider commenting out some of the datasets to avoid downloading them in main.")
                return
            target_file = os.path.join(dir, os.path.basename(link))
            try:
                download_file(link, dir) # zip file
            except requests.exceptions.RequestException as e:
                print(f"Error downloading file: {e}")
                return
            with zipfile.ZipFile(target_file, 'r') as zip_ref:
                for member in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
                    zip_ref.extract(member=member, path=dir)
            os.remove(os.path.join(dir, link.split('/')[-1]))

if __name__ == '__main__':
    with open(CONFIG_FILE, 'r') as f:
        cfg = yaml.safe_load(f)
    
    coco_data = COCO(cfg['COCO'])
    voc2012_data = VOC2012(cfg['VOC2012'])
    pcontext_data = PContext(cfg['pcontext'])
    adechallenge_data = ADEChallengeData2016(cfg['ADEChallengeData2016'])
    ade20k_data = ADE20K(cfg['ADE20K_2021_17_01'])

