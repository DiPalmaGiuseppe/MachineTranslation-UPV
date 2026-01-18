#!/bin/bash
#SBATCH -p docencia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --job-name=A2-NMs
#SBATCH -o logs/experiments_%j.log 
#SBATCH --mem=8G

cd ~/MachineTranslation-UPV/A2-CourseworkNeuralModels   # <-- vai nella directory giusta

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ~/envs/ta-project/

python -c "import torch; print('Python:', __import__('sys').executable); print('PyTorch version:', torch.__version__); print('CUDA disponibile:', torch.cuda.is_available())"

python -u experiments.py