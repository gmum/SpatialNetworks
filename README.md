# [Spatial Neural Networks](https://arxiv.org/abs/1910.02776)

## 1. Paper abstract ([arxiv](https://arxiv.org/abs/1910.02776))

We introduce bio-inspired artificial neural networks consisting of neurons that are additionally characterized by spatial positions. 
To simulate properties of biological systems we add the costs penalizing long connections and the proximity of neurons in a two-dimensional space. 
Our experiments show that in the case where the network performs two different tasks, the neurons naturally split into clusters, 
where each cluster is responsible for processing a different task. This behavior 
not only corresponds to the biological systems, but also allows for further insight into interpretability or continual learning. 

## 2. Dependencies

To start, install `conda` on your workstation and run

```
./install
```

from your root directory. This script will create separate `conda` environment
and install all necessary dependencies.

If the following doesn't work, you can install all dependencies via `pip`:

```shell
pip install -r requirements.txt
```

## 3. Performing experiments

### 3.1 Perform all steps [WIP]

Run `./experiment <args>` with appropriate arguments. All specific steps 
(if applicable to your settings will be performed) will be performed automatically.
Experiment results will be logged to appropriate file as well via [`DVC`](https://dvc.org/) 
(machine learning version control system).

You can find experiment related stuff inside your `cwd` within experiment.
Each experiment will be named by the varying arguments you passed, for example:

```

```

Available arguments are (should be specified in order):

- 
- 
- 
- 

### 3.2 Run specific step

Experiments are divided into subsections.
To perform specific part use `python main.py <subsection>`.

Currently following options are available

- `train` - train neural network
- `record` - record per task activations of neural network for later user
- `plot` - plot spatial locations of each layer
- `split` - split networks into task-specific subnetworks via some method
- `score` - score each network on specific task

Issue `python main.py <subsection> --help` to see available options for each subsection.


## 4. Replicating experiments [WIP]

[`DVC`](https://dvc.org/) is used to keep track of experiments. It should be installed
inside `conda` environment after running `./install` described in Dependencies subsection.

Run from `/src`:

```
./replicate <arguments>
```

in order to perform replication of necessary steps (those are cached if already existing).


Available arguments are (should be specified in order):

- 
- 
- 
- 

## 5. TO-DO code wise

- Fix `einsum` (line `112`, file `/src/nn/loss.py`) and generalize it for convolution case (major)
- Verify implemented splitting methods (replicate original results) residing in `/src/options/split/` (major)
- Verify previous results (major)
- Add plotting inside `/src/options/plot.py`
- Fix activation splitting to use unified API (`/src/options/split/activations.py`) (medium)
- Fix documentation (minor)
- Add README.md blanks (minor)


## 6. TO-DO experiment wise

- Add weights prunning based on spatial location
- Task independent splitting method
