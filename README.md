# Machine learning course held at the *Workshop for accelerating multi-messenger astronomy using air shower observations* at ICRR, UTokyo

**We recommend to use the pre-installed environment on our GPU accelerated jupyterlab nodes that we provide during the course. In that case, there is no need to setup any software on your computers.**

## Prequisites and installation instructions

For running our training course after the workshop or during the workshop on your own hardware, you may follow the following recommendations.

### Software requirements

The course materials require working installations of:

- Python > 3.9 
- torch
- torchvision
- pyg (pytorch_geometric)
- h5py
- scikit-learn
- numpy/scipy/matplotlib
- cmdstanpy
- git
- jupyter

If you know how to get those working on your laptop/remote workstation, please install those and disregard the rest of the document.

### Hardware/OS requirements

- 16+GB RAM
- a few GBs of storage 
- accelerator: CUDA, Apple Silicon (potentially AMD, but never tested)
- ~>8GB GPU memory
- Linux (x86) or MacOS (ARM)
- Intel Macs are not supported (no GPU, no acceleration)

### Python environment

We will use `Miniconda` but other package managers, such as pip, are equally fine.


#### Linux + Miniconda + CUDA

Follow [these instructions](https://www.anaconda.com/docs/getting-started/miniconda/install) to install `Miniconda`.

1. Download the training materials, either by using `git` or by downloading the archive from our [GitHub Repository](https://github.com/TA-DNN/TAML2025_ICRR_handson#).

2. Recreate the conda environment
Download the provided environment file (taml2025_minimal_env_linux_cuda.yml). Open a terminal in the folder containing this file, then run::

    `conda env create -f taml2025_minimal_env_linux_cuda.yml`

3. Activate the environment:
    
    `conda activate taml2025env`

4. Verify Installation
Run

    `conda list`

You should see a list of installed packages.

#### Pip (Linux and Mac)

The installation via pip is more generic but might be more difficult to debug. A working `python` with `pip` and `virtualenv` packages the starting point.

1. Create and activate a new virtualenv, and add it to jupyter (to avoid damaging your previous environemnt)

    ```
    python -m venv taml2025env
    source taml2025env/bin/activate
    python -m ipykernel install --user --name=taml2025venv
    ```

2. Install all packages: 

    `pip install torch torchvision torch_geometric cmdstanpy arviz scikit-learn numpy scipy matplotlib h5py jupyter ipykernel`

3. Finilize the installation of cmdstan

    `python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"`

4. Check that packages are installed

    `pip list`

5. The venv can be reactivated by

    `source taml2025env/bin/activate`

6. Launch jupyter and run notebooks from with the TAML2025 folder by browsing the explorer on the left side

    `jupyter lab`

