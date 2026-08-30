# Environment Setup

The project has been tested with:

* Python 3.10.12
* PyTorch 1.13.1
* CUDA 11.6
* PyTorch Geometric 2.3.1

## 1. Create a Python 3.10 virtual environment

```bash
python3.10 -m venv venv
source venv/bin/activate
```

Verify that the correct Python version is being used:

```bash
python --version
```

The output should show Python 3.10.12.

## 2. Install PyTorch

Install PyTorch 1.13.1 built for CUDA 11.6:

```bash
python -m pip install torch==1.13.1+cu116 \
    --extra-index-url https://download.pytorch.org/whl/cu116
```

## 3. Install PyTorch Geometric extensions

Install the PyG extensions matching PyTorch 1.13 and CUDA 11.6:

```bash
python -m pip install \
    torch-scatter==2.1.1+pt113cu116 \
    torch-sparse==0.6.17+pt113cu116 \
    torch-cluster==1.6.1+pt113cu116 \
    torch-spline-conv==1.2.2+pt113cu116 \
    -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
```

## 4. Install project dependencies

Install the remaining project dependencies:

```bash
python -m pip install -r requirements.txt
```

The `requirements.txt` file contains the project's direct Python dependencies, such as `transformers`, `datasets`, `numpy`, `tqdm`, and `torch-geometric`.

## 5. Verify the installation

Check the PyTorch and CUDA versions:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda)"
```

Expected:

```text
PyTorch: 1.13.1+cu116
CUDA: 11.6
```

Check that PyTorch Geometric can be imported:

```bash
python -c "import torch_geometric, torch_scatter, torch_sparse, torch_cluster, torch_spline_conv; print('PyG OK')"
```

Check the remaining main dependencies:

```bash
python -c "import transformers, datasets, numpy; print('Dependencies OK')"
```

## Reproducibility

`requirements-lock.txt` contains a snapshot of all packages installed in the known-working environment.

It was generated with:

```bash
python -m pip freeze --local > requirements-lock.txt
```

Use `requirements.txt` for normal installation. The lock file is retained as a reference for reproducing or debugging the exact tested environment.
