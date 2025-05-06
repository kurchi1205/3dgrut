sudo apt-get install gcc-11 g++-11
CUDA_VERSION=12.8.1 ./install_env.sh 3dgrut_cuda12 WITH_GCC11
pip install kagglehub
pip install torch
pip install -e .
pip install -r requirements.txt
python nerf_data_download.py 