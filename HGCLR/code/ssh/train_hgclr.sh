#!/bin/bash
#SBATCH --job-name=train_hgclr           # Name of the job
#SBATCH --output=logs/hgclr_train_out_%j.txt   # Standard output log (%j = job ID)
#SBATCH --error=logs/hgclr_train_err_%j.txt    # Standard error log
#SBATCH --partition=cs05r                # Name of the GPU partition (check your cluster docs)
#SBATCH --gres=gpu:1                     # Request 1 GPU (usually more than enough for SciBERT)
#SBATCH --nodes=1                        # Run on a single node
#SBATCH --ntasks=1                       # Single task
#SBATCH --cpus-per-task=8                # CPU cores for data loading
#SBATCH --mem=64G                        
#SBATCH --time=12:00:00                  # Wall clock time limit (HH:MM:SS)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=terence.tan@diamond.ac.uk

# ============================================================
# Seed
# ============================================================
# Use the first command-line argument as the seed.
# If no argument is supplied, default to 42.
SEED=${1:-42}

# Print job info
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Seed: ${SEED}"

# Load the software environment
module purge
module load cuda/11.6                    # Match this to your PyTorch version
module load python/3.10

# Create working directory in fast local storage
WORK_DIR=/tmp/panet_${SLURM_JOB_ID}
mkdir -p "${WORK_DIR}"


# Set cache directories to local storage (prevents filling home directory)
export HF_HOME=${WORK_DIR}/cache
export TRANSFORMERS_CACHE=${WORK_DIR}/cache
export TORCH_HOME=${WORK_DIR}/cache
export HF_DATASETS_CACHE=${WORK_DIR}/cache
export TOKENIZERS_PARALLELISM=false

echo "Working directory: ${WORK_DIR}"
echo ""

PROJECT_DIR=/dls/tmp/fdp54928/panet_classifier/hgclr

# Activate virtual environment
source "${PROJECT_DIR}/venv/bin/activate"

# Run from project root
cd "${PROJECT_DIR}" || exit 1

# Allow Python to find packages under hgclr/
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# Print environment info
echo "Current directory: $(pwd)"
echo "PYTHONPATH: ${PYTHONPATH}"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU name: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"
echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda if torch.cuda.is_available() else "N/A")')"
nvidia-smi
echo ""


# Execute the training script
python ssh/train.py \
    --name "hgclr_seed_${SEED}" \
    --seed "${SEED}" \
    --data "panet"


# Deactivate virtual environment
deactivate

echo "Job completed at $(date)"
