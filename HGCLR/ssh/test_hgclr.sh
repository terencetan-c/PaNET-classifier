#!/bin/bash
#SBATCH --job-name=test_hgclr_seed_42            # Name of the job
#SBATCH --output=logs/hgclr_test_seed_42_out_%j.txt   # Standard output log (%j = job ID)
#SBATCH --error=logs/hgclr_test_seed_42_err_%j.txt    # Standard error log
#SBATCH --partition=cs05r                # Name of the GPU partition (check your cluster docs)
#SBATCH --gres=gpu:1                     # Request 1 GPU (usually more than enough for SciBERT)
#SBATCH --nodes=1                        # Run on a single node
#SBATCH --ntasks=1                       # Single task
#SBATCH --cpus-per-task=8                # CPU cores for data loading
#SBATCH --mem=64G                        # 64GB RAM (A100 has 40GB, this gives headroom)
#SBATCH --time=12:00:00                  # Wall clock time limit (HH:MM:SS)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=terence.tan@diamond.ac.uk

# Print job info
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load the software environment
module purge
module load cuda/10.2                    # Match this to your PyTorch version
module load python/3.9                   # Or your preferred version

# Create working directory in fast local storage (402GB available!)
WORK_DIR=/tmp/panet_${SLURM_JOB_ID}

# Set cache directories to local storage (prevents filling home directory)
export HF_HOME=${WORK_DIR}/cache
export TRANSFORMERS_CACHE=${WORK_DIR}/cache
export TORCH_HOME=${WORK_DIR}/cache
export HF_DATASETS_CACHE=${WORK_DIR}/cache
export TOKENIZERS_PARALLELISM=false

echo "Working directory: ${WORK_DIR}"
echo ""

# Activate virtual environment
source /dls/tmp/fdp54928/panet_classifier/hgclr/venv/bin/activate

# Print environment info
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU name: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"
echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda if torch.cuda.is_available() else "N/A")')"
echo "GPU name: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"
nvidia-smi
echo ""


# Execute the training script
python /dls/tmp/fdp54928/panet_classifier/hgclr/test.py \
    --name 'panet-hgclr_seed_42_experiment_1' \
    --extra '_macro' \


# Deactivate virtual environment
deactivate

echo "Job completed at $(date)"
