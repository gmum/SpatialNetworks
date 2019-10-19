# Roadmap

Things one can do can be divided into two branches:
- code - improving current codebase
- experiment - perform additional experiments or verification

Next to each task there is a subjective priority (major, medium or minor).

Help with anything listed below would be highly appreciated.

## Code

- Fix `einsum` (line `112`, file `/src/nn/loss.py`) and generalize it for convolution case (major)
- Add plotting inside `/src/options/plot.py` (medium)
- Fix activation splitting to use unified API (`/src/options/split/activations.py`) (medium)
- Finish `experiment` and `reproduce` (medium)
- Add README.md blanks (minor)
- Check and validate user's input more thoroughly


## Experiments

- Verify and replicate previous results (major)
- Verify implemented splitting methods (replicate original results) residing in `/src/options/split/` (major)
- Add weights prunning based on spatial location (outlier-like detection) (medium)
- Create task independent splitting method (e.g. network can be splitted
with __absolutely no knowledge__ about incoming tasks)
