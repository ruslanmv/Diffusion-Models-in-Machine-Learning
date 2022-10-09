#SageMaker Setup
source activate python3
cd SageMaker
git clone https://github.com/ruslanmv/Diffusion-Models-in-Machine-Learning.git
cd Diffusion-Models-in-Machine-Learning/
conda create -n diffusion python==3.8 -y
conda activate diffusion
pip install torch pytorch_lightning  imageio torchvision
conda install ipykernel -y
python -m ipykernel install --user --name diffusion --display-name "Python (diffusion)"
