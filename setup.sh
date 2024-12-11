git submodule init
git submodule update

conda create -n vfss -python=3.10
conda activate -n vfss

pip install -e ./AlphaCLIP
pip install git+https://github.com/facebookresearch/detectron2.git
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install -r SAN/requirements.txt
pip install -r requirements.txt
pip install -U timm
git clone https://github.com/zhanghang1989/detail-api
cd detail-api/PythonAPI
make
cd ../..
echo "Packages installed, proceed with dataset installation!\nConfigure datasets.yaml and run download_dataset.py"
