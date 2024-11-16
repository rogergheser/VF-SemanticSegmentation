import yaml
import os
import zipfile
import tarfile
import wget
from tqdm import tqdm

####################################################################
# TODO Change prints with logging
# TODO Test COCO dataset download
# TODO Add other datasets
####################################################################

CONFIG_FILE = 'datasets.yaml'

def fields_exist(cfg: dict)->bool:
    """
    Checks if all fields of the configuration files already exist.
    :param cfg: dictionary with the configuration of the dataset.
    Should be passed as cfg[dataset] e.g. cfg['COCO'].
    """
    for field in cfg:
        if not os.path.exists(cfg[field]['dir']):
            print('Current directory:\n{}'.format(os.getcwd()))
            print('Field {} does not exist'.format(field))
            return False
    return True

def download_file(url: str, destination: str):
    """
    Downloads an input file from a given URL to a specified destination.
    :param url: string, URL of the file to download
    :param destination: string, destination path to save the file
    """
    # Create dir if missing
    if not os.path.exists(os.path.dirname(destination)):
        os.makedirs(os.path.dirname(destination))
    try:
        wget.download(url, destination, bar=wget.bar_thermometer)
    except Exception as e:
        print('Error downloading file: {}'.format(e))
    
class COCO():
    def __init__(self, cfg: dict, download:bool=True):
        """
        :param cfg: dictionary with the configuration of the dataset. Should be passed as cfg['COCO'].
        :param download: boolean, if True, the dataset is downloaded
        """
        self.cfg = cfg
        self.download = download

        if not fields_exist(cfg):
            # consider checking for missing files and only downloading those
            if download:
                self._download_dataset()
            else:
                raise ValueError('Dataset not found')
        assert fields_exist(self.cfg), "Error downloading dataset"

    def _download_dataset(self):
        """
        Downloads files for the COCO dataset according to the provided configuration file.
        """
        for field in self.cfg:
            link, dir = self.cfg[field]['url'], self.cfg[field]['dir']
            if not os.path.exists(dir):
                download_file(link, dir)
                # All COCO download files are zip
                with zipfile.ZipFile(os.path.join(dir, link.split('/')[-1]), 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(dir))
                print("Successfully downloaded and extracted COCO:{}".format(field))
            else:
                print('Already exists. Skipping download\n{}\n{}'.format(link, dir))


class VOC2012():
    def __init__(self, cfg: dict, download:bool=True):
        """
        :param cfg: dictionary with the configuration of the dataset. Should be passed as cfg['VOC2012'].
        :param download: boolean, if True, the dataset is downloaded
        """
        self.cfg = cfg
        self.download = download

        if not fields_exist(cfg):
            # consider checking for missing files and only downloading those
            if download:
                self._download_dataset()
            else:
                raise ValueError('Dataset not found')
        assert fields_exist(self.cfg), "Error downloading dataset"

    def _download_dataset(self):
        """
        Downloads files for the VOC2012 dataset according to the provided configuration file.
        """
        link, dir = self.cfg['trainval']['url'], self.cfg['trainval']['dir']
        download_file(link, dir)
        with tarfile.open(os.path.join(dir, link.split('/')[-1]), 'r') as tar:
            # Go over each member
            for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
                # Extract member
                tar.extract(member=member)
        # TODO test this
        # ASSERT DOWNLOADED FILES ARE THERE OR PROMPT THE USER TO GET THEM

class PContext():
    def __init__(self, cfg: dict, download:bool=True):
        self.cfg = cfg
        self.download = download

        if not fields_exist(cfg):
            # consider checking for missing files and only downloading those
            if download:
                self._download_dataset()
            else:
                raise ValueError('Dataset not found')

    def download_dataset(self):
        link, dir = self.cfg['images']['url'], self.cfg['images']['dir']
        download_file(link, dir) # tar file
        with tarfile.open(os.path.join(dir, link.split('/')[-1]), 'r') as tar:
            # Go over each member
            for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
                # Extract member
                tar.extract(member=member)

        link, dir = self.cfg['labels']['url'], self.cfg['labels']['dir']
        download_file(link, dir) # json file

class ADEChallengeData2016():
    def __init__(self, cfg: dict, download:bool=True):
        self.cfg = cfg
        self.download = download

        if not fields_exist(cfg):
            # consider checking for missing files and only downloading those
            if download:
                self._download_dataset()
            else:
                raise ValueError('Dataset not found')

    def download_dataset(self):
        link, dir = self.cfg['images']['url'], self.cfg['images']['dir']
        download_file(link, dir) # zip file
        with zipfile.ZipFile(os.path.join(dir, link.split('/')[-1]), 'r') as zip_ref:
            for member in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
                zip_ref.extract(member=member)


if __name__ == '__main__':
    with open(CONFIG_FILE, 'r') as f:
        cfg = yaml.safe_load(f)

    coco_data = COCO(cfg['COCO'])
    voc2012_data = VOC2012(cfg['VOC2012'])
    pcontext_data = PContext(cfg['PContext'])
    adechallenge_data = ADEChallengeData2016(cfg['ADEChallengeData2016'])

