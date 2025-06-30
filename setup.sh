
apt-get update && apt-get install nano zip ffmpeg libsm6 libxext6 unar vim htop unzip gcc curl g++ python3-distutils python3-apt -y

#conda create --name ConceptDiff python=3.10
pip install -r requirements.txt

python3 setup.py build_ext --inplace
python setup_sparse_cy_tste.py build_ext -i

pip install git+https://github.com/openai/CLIP.git