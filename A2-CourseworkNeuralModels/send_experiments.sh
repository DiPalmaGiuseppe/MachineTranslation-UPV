#!/bin/bash
#SBATCH -p docencia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --job-name=A2-NMs
#SBATCH -o logs/experiments_%j.log 
#SBATCH --mem=8G

# -------------------------
# Check input argument
# -------------------------
if [ -z "$1" ]; then
    echo "ERROR: No experiment specified."
    echo "Usage: sbatch send_experiments.sh [nllb|llama|opt]"
    exit 1
fi

EXPERIMENT=$1

cd ~/MachineTranslation-UPV/A2-CourseworkNeuralModels || exit 1

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ~/envs/ta-project/

echo "Running experiment: $EXPERIMENT"
python -c "import torch; print('Python:', __import__('sys').executable); print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# -------------------------
# Select experiment
# -------------------------
if [ "$EXPERIMENT" = "nllb" ]; then
    python -u experiments_nllb.py

elif [ "$EXPERIMENT" = "llama" ]; then
    python -u experiments_llama.py

elif [ "$EXPERIMENT" = "opt" ]; then
    python -u experiments_opt.py

else
    echo "ERROR: Unknown experiment '$EXPERIMENT'"
    echo "Valid options: nllb | llama | opt"
    exit 1
fi