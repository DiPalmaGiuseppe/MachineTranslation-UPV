#!/bin/bash
#SBATCH -p docencia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --job-name=A3-ST
#SBATCH -o logs/experiments_%j.log 
#SBATCH --mem=8G

# -------------------------
# Check input argument
# -------------------------
if [ -z "$1" ]; then
    echo "ERROR: No experiment specified."
    echo "Usage: sbatch send_experiments.sh [asr_baseline|st_baseline|st_cascade_baseline|st_cascade_finetuned]"
    exit 1
fi

EXPERIMENT=$1

cd ~/MachineTranslation-UPV/A3-CourseworkSpeechTranslation || exit 1

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ~/envs/ta-project/

echo "Running experiment: $EXPERIMENT"
python -c "import torch; print('Python:', __import__('sys').executable); print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# -------------------------
# Select experiment
# -------------------------
if [ "$EXPERIMENT" = "asr_baseline" ]; then
    python -u ASR_Baseline.py

elif [ "$EXPERIMENT" = "st_baseline" ]; then
    python -u ST_Baseline.py
elif [ "$EXPERIMENT" = "st_cascade_baseline" ]; then
    echo "TODO"
elif [ "$EXPERIMENT" = "st_cascade_finetuned" ]; then
    echo "TODO"

else
    echo "ERROR: Unknown experiment '$EXPERIMENT'"
    echo "Valid options: asr_baseline | st_baseline | st_cascade_baseline | st_cascade_finetuned"
    exit 1
fi