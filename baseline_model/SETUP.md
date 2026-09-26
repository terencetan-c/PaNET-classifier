## Environment Setup

The experiments were run using Python 3.11 on an NVIDIA A100-SXM4-40GB GPU.

### Software environment

* Python: **3.11**
* PyTorch: **2.5.1+cu121**
* PyTorch CUDA build: **CUDA 12.1**
* GPU: **NVIDIA A100-SXM4-40GB**
* BF16: supported
* TF32: enabled by the training script

The PyTorch installation uses the CUDA 12.1 wheel. A system CUDA module is **not required for normal execution of this code**; the PyTorch wheel provides the required CUDA runtime libraries.

### 1. Create the Python environment

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 2. Install PyTorch

Install the CUDA 12.1 build of PyTorch:

```bash
python -m pip install --upgrade pip

python -m pip install torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install the remaining dependencies

Install the versions used for the experiments:

```bash
python -m pip install \
    transformers==4.46.3 \
    datasets==3.1.0 \
    accelerate==1.1.1 \
    pandas==2.2.3 \
    numpy==1.26.4 \
    scikit-learn==1.5.2 \
    fastparquet==2024.11.0 \
    pyarrow==18.1.0
```

Alternatively, install the dependencies from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Note that `torch` is installed separately because the CUDA-specific PyTorch wheel is required.

### 4. Verify the installation

Run the following on a GPU allocation:

```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA build:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
print('BF16 supported:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else 'N/A')
"
```

Expected output:

```text
PyTorch: 2.5.1+cu121
CUDA build: 12.1
CUDA available: True
GPU: NVIDIA A100-SXM4-40GB
BF16 supported: True
```

### 5. GPU allocation

The training requires an NVIDIA GPU because mixed-precision BF16 training is enabled in the script.

On a Slurm-based cluster, request a GPU allocation using the appropriate GPU partition for the cluster. For example:

```bash
srun --partition=<GPU_PARTITION> --gres=gpu:1 --pty bash
```

The exact partition name is cluster-specific.

Verify that a GPU is available before running the experiment:

```bash
nvidia-smi
```

### 6. Run the experiment

After activating the environment and obtaining a GPU allocation:

```bash
source my_venv/bin/activate
python <training_script>.py
```

The experiment uses five random seeds:

```python
seeds = [42, 1, 2, 3, 4]
```

For an initial smoke test, this can temporarily be changed to:

```python
seeds = [42]
```

before running the full five-seed experiment.

### Reproducibility

`requirements-lock.txt` contains a snapshot of all packages installed in the known-working environment.

It was generated with:

```bash
python -m pip freeze --local > requirements-lock.txt
```

Use `requirements.txt` for normal installation. The lock file is retained as a reference for reproducing or debugging the exact tested environment.



The experiment explicitly sets the random seed using Hugging Face Transformers' `set_seed()` function. The training script also enables TF32 for CUDA matrix multiplication and cuDNN before training:

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

The training configuration uses BF16 mixed precision:

```python
fp16=False
bf16=True
```

These settings assume an A100-class GPU.
