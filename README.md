# [Spatial Neural Networks](https://arxiv.org/abs/1910.02776)

| Version | Docs | Style | Python | PyTorch | Contribute | Roadmap |
|---------|------|-------|--------|---------|------------|---------|
| [![Version](https://img.shields.io/static/v1?label=&message=0.0.1&color=377EF0&style=for-the-badge)](https://arxiv.org/abs/1910.02776) | [![Documentation](https://img.shields.io/static/v1?label=&message=docs&color=EE4C2C&style=for-the-badge)](TBD)  | [![style](https://img.shields.io/static/v1?label=&message=CB&color=27A8E0&style=for-the-badge)](TBD) | [![Python](https://img.shields.io/static/v1?label=&message=3.7&color=377EF0&style=for-the-badge&logo=python&logoColor=F8C63D)](https://www.python.org/) | [![PyTorch](https://img.shields.io/static/v1?label=&message=1.2.0&color=EE4C2C&style=for-the-badge)](https://pytorch.org/) | [![Contribute](https://img.shields.io/static/v1?label=&message=guide&color=009688&style=for-the-badge)](https://github.com/szymonmaszke/torchdata/blob/master/CONTRIBUTING.md) | [![Roadmap](https://img.shields.io/static/v1?label=&message=roadmap&color=f50057&style=for-the-badge)](https://github.com/szymonmaszke/torchdata/blob/master/ROADMAP.md)

## 1. Paper abstract ([arxiv](https://arxiv.org/abs/1910.02776))

We introduce bio-inspired artificial neural networks consisting of neurons that are additionally characterized by spatial positions. 
To simulate properties of biological systems we add the costs penalizing long connections and the proximity of neurons in a two-dimensional space. 
Our experiments show that in the case where the network performs two different tasks, the neurons naturally split into clusters, 
where each cluster is responsible for processing a different task. This behavior 
not only corresponds to the biological systems, but also allows for further insight into interpretability or continual learning. 

## 2. Dependencies

Dependencies are gathered inside `requirements.txt`.
We advise to use `conda` environment for easier package management.

### 2.1 Setup `conda` [optional]

- Install conda for your specific OS, see instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
- Create new environment by issuing from shell: `$ conda create --name SpatialNetworks`
- Activate environment: `$ conda activate SpatialNetworks`
- Install `pip` within environment: `$ conda install pip`

### 2.2 Install packages

Make sure you have `pip` installed (see [documentation](https://packaging.python.org/tutorials/installing-packages/#ensure-you-can-run-pip-from-the-command-line)) and run:

```
pip install -r requirements.txt
```

Specify `--user` flag if needed.

## 3. Performing experiments

Experiments are divided into subsections.
To perform specific part use `python main.py <subsection>`.

Currently following options are available

- `train` - train neural network
- `record` - record per task activations of neural network for later user
- `plot` - plot spatial locations of each layer
- `split` - split networks into task-specific subnetworks via some method
- `score` - score each network on specific task

Issue `python main.py <subsection> --help` to see available options for each subsection.

To help with reproducibility later, please wrap your experiments commands with `dvc` (see their [documentation](https://dvc.org/doc)).
