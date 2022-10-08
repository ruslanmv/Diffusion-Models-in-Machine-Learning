#SageMaker Setup
source activate python3
cd Sagemaker
git clone https://github.com/ruslanmv/Diffusion-Models-in-Machine-Learning.git
cd Diffusion-Models-in-Machine-Learning/
conda create -n difussion python=3.8
conda activate difussion
pip install  pytorch pytorch_lightning  imageio torchvision
